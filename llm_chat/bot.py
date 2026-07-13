import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

from llm_chat.model import LLMClient, EXPERIMENT_CONFIGS

# Настройка логирования
from llm_chat.token_tracker import TokenTracker

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class GenerationParams:
    """Параметры генерации для пользователя."""
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    
    def update(self, param: str, delta: float, max_val: float) -> None:
        """Обновить параметр с проверкой границ."""
        current = getattr(self, param)
        new_value = max(0.0, min(max_val, current + delta))
        
        # Округляем для температуры и top_p
        if param in ("temperature", "top_p"):
            new_value = round(new_value, 1)
        elif param == "top_k":
            new_value = int(new_value)
        
        setattr(self, param, new_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать в словарь."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class LLMBot:
    """
    Telegram-бот для взаимодействия с локальной LLM.
    
    Возможности:
    - Обычный чат с моделью
    - Настройка параметров генерации через кнопки
    - Режим сравнения (эксперимент)
    - Обработка долгих запросов в отдельных потоках
    - Логирование ошибок
    """
    
    # Параметры для больших моделей
    BIG_MODEL_OPTIONS = {
        "num_predict": 512,      # Максимальная длина ответа в токенах
        "num_ctx": 4096,         # Размер контекстного окна
    }
    
    # Пресеты для быстрой смены режимов
    PRESETS = {
        "code": GenerationParams(temperature=0.2, top_k=20, top_p=0.5),
        "balanced": GenerationParams(temperature=0.7, top_k=40, top_p=0.9),
        "creative": GenerationParams(temperature=1.2, top_k=60, top_p=0.95),
    }
    
    def __init__(
        self,
        token: Optional[str] = None,
        model_name: Optional[str] = None,
        max_workers: int = 2,
        request_timeout: int = 120
    ):
        """
        Инициализация бота.
        
        Args:
            token: Токен Telegram бота
            model_name: Название модели в Ollama
            max_workers: Количество потоков для параллельных запросов
            request_timeout: Таймаут на запрос к модели (сек)
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        self.request_timeout = request_timeout
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN не задан")
        
        # Клиент для работы с Ollama
        self.llm_client = LLMClient(model=self.model_name)

        self.token_tracker = TokenTracker(
            log_dir="token_logs",
            model=self.model_name
        )

        # Состояние пользователей
        self.user_settings: Dict[int, GenerationParams] = {}
        self.user_experiment_mode: Dict[int, bool] = {}
        
        # Пул потоков для CPU-интенсивных операций
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Приложение Telegram
        self.app = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики
        self._register_handlers()
        
        logger.info(f"Бот инициализирован с моделью: {self.model_name}")
    
    # ========================================================================
    # Регистрация обработчиков
    # ========================================================================
    
    def _register_handlers(self) -> None:
        """Зарегистрировать все обработчики команд и сообщений."""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("settings", self.settings))
        self.app.add_handler(CommandHandler("experiment", self.experiment))
        self.app.add_handler(CommandHandler("reset", self.reset))
        self.app.add_handler(CommandHandler("model", self.show_model_info))
        self.app.add_handler(CommandHandler("preset", self.set_preset))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(
            CallbackQueryHandler(self.button_handler, pattern="^(temp|topk|topp)_")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.preset_button_handler, pattern="^preset_")
        )
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Обработчик ошибок
        self.app.add_error_handler(self.error_handler)
    
    # ========================================================================
    # Обработчики команд
    # ========================================================================
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /start."""
        user_id = update.effective_user.id
        self._init_user(user_id)
        
        await update.message.reply_text(
            "🤖 *Локальный LLM-бот*\n\n"
            f"Модель: `{self.model_name}`\n"
            "Запущена через Ollama на вашем компьютере\n\n"
            "📝 Просто напишите вопрос\n"
            "⚙️ /settings — настройки генерации\n"
            "🧪 /experiment — сравнить 3 режима\n"
            "🎭 /preset — быстрые пресеты (code, balanced, creative)\n"
            "ℹ️ /model — информация о модели\n"
            "🔄 /reset — сбросить настройки",
            parse_mode="Markdown"
        )
    
    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Показать настройки с кнопками изменения."""
        user_id = update.effective_user.id
        params = self._get_user_params(user_id)
        
        keyboard = self._build_settings_keyboard(params)
        
        await update.message.reply_text(
            "⚙️ *Настройки генерации*\n"
            "_Нажимайте кнопки для изменения:_",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def experiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Запуск режима сравнения параметров."""
        user_id = update.effective_user.id
        self.user_experiment_mode[user_id] = True
        
        await update.message.reply_text(
            "🧪 *Режим сравнения параметров*\n\n"
            "Отправьте промпт, и я покажу 3 варианта:\n"
            f"• 🎯 Точный (T=0.2, K=10, P=0.5)\n"
            f"• ⚖️ Сбалансированный (T=0.7, K=40, P=0.9)\n"
            f"• 🎨 Креативный (T=1.5, K=80, P=1.0)\n\n"
            "Для выхода: /reset",
            parse_mode="Markdown"
        )

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Показать статистику использования токенов."""
        summary = self.token_tracker.format_summary()
        await update.message.reply_text(summary, parse_mode="Markdown")

    async def show_model_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Показать информацию о текущей модели."""
        await update.message.reply_text(
            f"ℹ️ *Информация о модели*\n\n"
            f"Название: `{self.model_name}`\n"
            f"Таймаут запроса: {self.request_timeout} сек\n"
            f"Максимум токенов ответа: {self.BIG_MODEL_OPTIONS['num_predict']}\n"
            f"Размер контекста: {self.BIG_MODEL_OPTIONS['num_ctx']}\n\n"
            "Для смены модели измените OLLAMA_MODEL в .env",
            parse_mode="Markdown"
        )
    
    async def set_preset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Показать пресеты для быстрой настройки."""
        keyboard = [
            [
                InlineKeyboardButton("💻 Code (T=0.2)", callback_data="preset_code"),
                InlineKeyboardButton("⚖️ Balanced (T=0.7)", callback_data="preset_balanced"),
            ],
            [
                InlineKeyboardButton("🎨 Creative (T=1.2)", callback_data="preset_creative"),
            ],
        ]
        
        await update.message.reply_text(
            "🎭 *Выберите пресет:*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Сброс настроек и режимов."""
        user_id = update.effective_user.id
        self._init_user(user_id)
        await update.message.reply_text("✅ Настройки сброшены до стандартных")
    
    # ========================================================================
    # Обработчики кнопок
    # ========================================================================
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик кнопок изменения параметров."""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        params = self._get_user_params(user_id)
        
        # Карта изменений параметров
        changes = {
            "temp_up": ("temperature", 0.1, 2.0),
            "temp_down": ("temperature", -0.1, 2.0),
            "topk_up": ("top_k", 10, 100),
            "topk_down": ("top_k", -10, 100),
            "topp_up": ("top_p", 0.1, 1.0),
            "topp_down": ("top_p", -0.1, 1.0),
        }
        
        if query.data in changes:
            param, delta, max_val = changes[query.data]
            params.update(param, delta, max_val)
            self.user_settings[user_id] = params
        
        await query.edit_message_reply_markup(
            reply_markup=InlineKeyboardMarkup(self._build_settings_keyboard(params))
        )
    
    async def preset_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик кнопок выбора пресета."""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        preset_name = query.data.replace("preset_", "")
        
        if preset_name in self.PRESETS:
            self.user_settings[user_id] = self.PRESETS[preset_name]
            params = self.PRESETS[preset_name]
            
            await query.edit_message_text(
                f"✅ Пресет *{preset_name}* применён\n"
                f"T={params.temperature}, K={params.top_k}, P={params.top_p}",
                parse_mode="Markdown"
            )
    
    # ========================================================================
    # Основной обработчик сообщений
    # ========================================================================
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик текстовых сообщений."""
        user_id = update.effective_user.id
        text = update.message.text
        
        # Режим эксперимента
        if self.user_experiment_mode.get(user_id, False):
            await self._handle_experiment(update, text, user_id)
            return
        
        # Обычный режим
        await self._handle_regular_message(update, text, user_id)
    
    async def _handle_experiment(
        self, update: Update, text: str, user_id: int
    ) -> None:
        """Обработка сообщения в режиме эксперимента."""
        self.user_experiment_mode[user_id] = False
        
        status_msg = await update.message.reply_text(
            "🧪 Провожу эксперимент...\n"
            "Генерирую 3 варианта ответа"
        )
        
        try:
            # Генерация в отдельном потоке
            loop = asyncio.get_event_loop()
            
            def run_experiment():
                return self.llm_client.compare_parameters(text, EXPERIMENT_CONFIGS)
            
            results = await asyncio.wait_for(
                loop.run_in_executor(self.executor, run_experiment),
                timeout=self.request_timeout * 3  # В 3 раза больше для эксперимента
            )
            
            await status_msg.delete()
            
            for result in results:
                self.token_tracker.log(
                    prompt=text,
                    response=result['answer'],
                    duration=result.get('duration', 0),
                    temperature=result['params']['temperature'],
                    top_k=result['params']['top_k'],
                    top_p=result['params']['top_p'],
                    user_id=user_id,
                )
                response_text = (
                    f"*{result['label']}*\n"
                    f"`T={result['params']['temperature']}, "
                    f"K={result['params']['top_k']}, "
                    f"P={result['params']['top_p']}`\n\n"
                    f"{result['answer'][:1000]}"
                )
                await update.message.reply_text(response_text, parse_mode="Markdown")
                
        except asyncio.TimeoutError:
            await status_msg.edit_text(
                "⏰ Эксперимент занял слишком много времени.\n"
                "Попробуйте более короткий промпт или другую модель."
            )
        except Exception as e:
            await status_msg.edit_text(f"❌ Ошибка эксперимента: {str(e)[:200]}")
            logger.error(f"Experiment error for user {user_id}: {e}")

    async def _handle_regular_message(
            self, update: Update, text: str, user_id: int
    ) -> None:
        """Обработка обычного сообщения."""
        params = self._get_user_params(user_id)

        status_msg = await update.message.reply_text(
            f"⏳ Генерирую ответ...\n"
            f"Модель: `{self.model_name}`\n"
            f"Параметры: T={params.temperature}, K={params.top_k}, P={params.top_p}"
        )

        try:
            loop = asyncio.get_event_loop()

            def generate():
                # Засекаем время
                start = time.time()
                answer = self.llm_client.chat(
                    prompt=text,
                    temperature=params.temperature,
                    top_k=params.top_k,
                    top_p=params.top_p,
                )
                elapsed = time.time() - start
                return answer, elapsed

            answer, duration = await asyncio.wait_for(
                loop.run_in_executor(self.executor, generate),
                timeout=self.request_timeout
            )

            # === Логируем использование токенов ===
            usage = self.token_tracker.log(
                prompt=text,
                response=answer,
                duration=duration,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                user_id=user_id,
            )

            await status_msg.delete()

            token_info = (
                f"\n\n📊 _Запрос: {usage['prompt_tokens']} токенов | "
                f"Ответ: {usage['response_tokens']} токенов | "
                f"Время: {usage['duration_seconds']}с_"
            )

            full_response = answer + token_info

            await self._send_long_message(update, full_response)

        except asyncio.TimeoutError:
            await status_msg.edit_text("⏰ Генерация заняла слишком много времени...")
        except Exception as e:
            await status_msg.edit_text(f"❌ Ошибка: {str(e)[:200]}")
            logger.error(f"Generation error for user {user_id}: {e}")

    # ========================================================================
    # Вспомогательные методы
    # ========================================================================
    
    def _init_user(self, user_id: int) -> None:
        """Инициализировать настройки пользователя."""
        self.user_settings[user_id] = GenerationParams()
        self.user_experiment_mode[user_id] = False
    
    def _get_user_params(self, user_id: int) -> GenerationParams:
        """Получить параметры пользователя (с инициализацией при необходимости)."""
        if user_id not in self.user_settings:
            self._init_user(user_id)
        return self.user_settings[user_id]
    
    def _build_settings_keyboard(self, params: GenerationParams) -> list:
        """Создать клавиатуру настроек."""
        return [
            [
                InlineKeyboardButton("🔥 -", callback_data="temp_down"),
                InlineKeyboardButton(f"T: {params.temperature:.1f}", callback_data="temp_show"),
                InlineKeyboardButton("🔥 +", callback_data="temp_up"),
            ],
            [
                InlineKeyboardButton("🎯 -", callback_data="topk_down"),
                InlineKeyboardButton(f"K: {params.top_k}", callback_data="topk_show"),
                InlineKeyboardButton("🎯 +", callback_data="topk_up"),
            ],
            [
                InlineKeyboardButton("🎲 -", callback_data="topp_down"),
                InlineKeyboardButton(f"P: {params.top_p:.1f}", callback_data="topp_show"),
                InlineKeyboardButton("🎲 +", callback_data="topp_up"),
            ],
        ]
    
    async def _send_long_message(
        self, update: Update, text: str, max_length: int = 4000
    ) -> None:
        """Отправить длинное сообщение, разбив на части."""
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return
        
        # Разбиваем на части
        parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                await update.message.reply_text(part)
            else:
                await update.message.reply_text(f"{part}\n\n_(продолжение следует...)_")
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Глобальный обработчик ошибок."""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Произошла внутренняя ошибка. Попробуйте позже или используйте /reset"
            )
    
    # ========================================================================
    # Запуск и остановка
    # ========================================================================
    
    async def health_check(self) -> bool:
        """Проверить работоспособность модели перед запуском."""
        test_prompts = ["Hello", "def test(): pass"]
        
        for prompt in test_prompts:
            try:
                loop = asyncio.get_event_loop()
                
                def test_generate():
                    return self.llm_client.chat(
                        prompt=prompt,
                        temperature=0.1,
                        top_k=1,
                        top_p=0.1,
                    )
                
                await asyncio.wait_for(
                    loop.run_in_executor(self.executor, test_generate),
                    timeout=30
                )
                logger.info(f"✅ Health check passed: '{prompt}'")
            except Exception as e:
                logger.error(f"❌ Health check failed for '{prompt}': {e}")
                return False
        
        return True
    
    def run(self) -> None:
        """Запустить бота."""
        logger.info(f"Запуск бота с моделью: {self.model_name}")
        
        # Проверяем здоровье модели
        loop = asyncio.get_event_loop()
        if not loop.run_until_complete(self.health_check()):
            logger.error("Health check не пройден! Проверьте Ollama и модель.")
            return
        
        # Запускаем бота
        self.app.run_polling()
    
    def shutdown(self) -> None:
        """Корректно остановить бота."""
        logger.info("Остановка бота...")
        self.executor.shutdown(wait=True)


def main():
    """Точка входа для poetry scripts."""
    try:
        bot = LLMBot(
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120"))
        )
        bot.run()
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()