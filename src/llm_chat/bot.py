"""Telegram-бот для локальной LLM."""

import asyncio
import os
from typing import Dict, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

from .model import LLMClient, EXPERIMENT_CONFIGS

load_dotenv()

# Конфигурация
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# Глобальные объекты
llm_client = LLMClient(model=MODEL_NAME)
user_settings: Dict[int, Dict[str, Any]] = {}

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user_id = update.effective_user.id
    user_settings[user_id] = DEFAULT_PARAMS.copy()

    await update.message.reply_text(
        "🤖 *Локальный LLM-бот*\n\n"
        f"Модель: `{MODEL_NAME}`\n"
        "Запущена через Ollama на вашем компьютере\n\n"
        "📝 Просто напишите вопрос\n"
        "⚙️ /settings — настройки генерации\n"
        "🧪 /experiment — сравнить 3 режима\n"
        "🔄 /reset — сбросить настройки",
        parse_mode="Markdown"
    )


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать настройки с кнопками изменения."""
    user_id = update.effective_user.id
    params = user_settings.get(user_id, DEFAULT_PARAMS)

    keyboard = _build_settings_keyboard(params)

    await update.message.reply_text(
        "⚙️ *Настройки генерации*\n"
        "_Нажимайте кнопки для изменения:_",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


def _build_settings_keyboard(params: Dict[str, Any]) -> list:
    """Создать клавиатуру настроек."""
    return [
        [
            InlineKeyboardButton("🔥 -", callback_data="temp_down"),
            InlineKeyboardButton(f"T: {params['temperature']:.1f}", callback_data="temp_show"),
            InlineKeyboardButton("🔥 +", callback_data="temp_up"),
        ],
        [
            InlineKeyboardButton("🎯 -", callback_data="topk_down"),
            InlineKeyboardButton(f"K: {params['top_k']}", callback_data="topk_show"),
            InlineKeyboardButton("🎯 +", callback_data="topk_up"),
        ],
        [
            InlineKeyboardButton("🎲 -", callback_data="topp_down"),
            InlineKeyboardButton(f"P: {params['top_p']:.1f}", callback_data="topp_show"),
            InlineKeyboardButton("🎲 +", callback_data="topp_up"),
        ],
    ]


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок настроек."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    params = user_settings.get(user_id, DEFAULT_PARAMS)

    # Изменение параметров
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
        params[param] = max(0.0, min(max_val, params[param] + delta))
        if param in ("temperature", "top_p"):
            params[param] = round(params[param], 1)

    user_settings[user_id] = params
    await query.edit_message_reply_markup(
        reply_markup=InlineKeyboardMarkup(_build_settings_keyboard(params))
    )


async def experiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запуск режима эксперимента."""
    context.user_data["awaiting_experiment"] = True

    await update.message.reply_text(
        "🧪 *Режим сравнения параметров*\n\n"
        "Отправьте промпт, и я покажу 3 варианта:\n"
        f"• 🎯 Точный (T=0.2, K=10, P=0.5)\n"
        f"• ⚖️ Сбалансированный (T=0.7, K=40, P=0.9)\n"
        f"• 🎨 Креативный (T=1.5, K=80, P=1.0)\n\n"
        "Для выхода: /reset",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений."""
    user_id = update.effective_user.id
    text = update.message.text

    # Режим эксперимента
    if context.user_data.get("awaiting_experiment"):
        context.user_data["awaiting_experiment"] = False

        status_msg = await update.message.reply_text("🧪 Провожу эксперимент...")

        # Используем метод из model.py
        results = llm_client.compare_parameters(text, EXPERIMENT_CONFIGS)

        await status_msg.delete()

        for result in results:
            response_text = (
                f"*{result['label']}*\n"
                f"`T={result['params']['temperature']}, "
                f"K={result['params']['top_k']}, "
                f"P={result['params']['top_p']}`\n\n"
                f"{result['answer'][:1000]}"
            )
            await update.message.reply_text(response_text, parse_mode="Markdown")
        return

    # Обычный режим
    params = user_settings.get(user_id, DEFAULT_PARAMS)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        answer = llm_client.chat(
            prompt=text,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
        )

        # Telegram ограничение на длину сообщения
        for i in range(0, len(answer), 4000):
            await update.message.reply_text(answer[i:i + 4000])

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сброс настроек и режимов."""
    user_id = update.effective_user.id
    user_settings[user_id] = DEFAULT_PARAMS.copy()
    context.user_data.pop("awaiting_experiment", None)
    await update.message.reply_text("✅ Настройки сброшены")


def main():
    """Точка входа для poetry scripts."""
    app = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings", settings))
    app.add_handler(CommandHandler("experiment", experiment))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(button_handler, pattern="^(temp|topk|topp)_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"🤖 Бот запущен с моделью: {MODEL_NAME}")
    app.run_polling()


if __name__ == "__main__":
    main()