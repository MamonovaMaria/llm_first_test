# src/llm_chat/console_app.py
"""Консольный интерфейс AI-агента."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from llm_chat.model import LLMClient
from llm_chat.token_tracker import TokenTracker
from llm_chat.agent import AIAgent

load_dotenv()


class ConsoleApp:
    """Консольный интерфейс для AI-агента."""

    # Цвета для вывода (ANSI)
    COLORS = {
        "reset": "\033[0m",
        "user": "\033[94m",  # Синий
        "assistant": "\033[92m",  # Зелёный
        "system": "\033[93m",  # Жёлтый
        "error": "\033[91m",  # Красный
        "info": "\033[96m",  # Голубой
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    def __init__(
            self,
            model_name: Optional[str] = None,
            project_path: str = ".",
    ):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        self.project_path = Path(project_path).resolve()

        # Инициализация компонентов
        self.llm_client = LLMClient(model=self.model_name)
        self.token_tracker = TokenTracker(model=self.model_name)
        self.agent = AIAgent(
            llm_client=self.llm_client,
            token_tracker=self.token_tracker,
        )

        # Настройка ProjectReader на нужную директорию
        self.agent.project_reader.base_path = self.project_path

        # Счётчик сообщений
        self.message_count = 0

    def _print(self, text: str, color: str = "reset", end: str = "\n") -> None:
        """Вывести цветной текст."""
        print(f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}", end=end)

    def _print_banner(self) -> None:
        """Вывести приветственный баннер."""
        print()
        self._print("╔══════════════════════════════════════════════╗", "info")
        self._print("║         🤖 Локальный AI-агент               ║", "bold")
        self._print("╠══════════════════════════════════════════════╣", "info")
        self._print(f"║ Модель: {self.model_name:<36s}║", "info")
        self._print(f"║ Проект: {str(self.project_path)[:36]:<36s}║", "info")
        self._print("╠══════════════════════════════════════════════╣", "info")
        self._print("║ Команды:                                    ║", "info")
        self._print("║ /read <путь>   - прочитать файл/проект      ║", "info")
        self._print("║ /file <путь>   - прочитать файл             ║", "info")
        self._print("║ /clear         - очистить историю           ║", "info")
        self._print("║ /memory        - статистика памяти          ║", "info")
        self._print("║ /system <текст>- задать системный промпт    ║", "info")
        self._print("║ /stats         - статистика токенов         ║", "info")
        self._print("║ /help          - помощь                     ║", "info")
        self._print("║ /exit          - выход                      ║", "info")
        self._print("╚══════════════════════════════════════════════╝", "info")
        print()

    def _print_help(self) -> None:
        """Вывести справку."""
        print()
        self._print("📋 Доступные команды:", "bold")
        print()
        self._print("  /read <путь>      ", "user", end="")
        print("- прочитать файл или проект (пример: /read .)")
        self._print("  /file <путь>      ", "user", end="")
        print("- прочитать конкретный файл (пример: /file src/main.py)")
        self._print("  /clear            ", "user", end="")
        print("- очистить историю диалога")
        self._print("  /memory           ", "user", end="")
        print("- показать статистику памяти")
        self._print("  /system <текст>   ", "user", end="")
        print("- задать системный промпт")
        self._print("  /stats            ", "user", end="")
        print("- показать статистику токенов")
        self._print("  /help             ", "user", end="")
        print("- эта справка")
        self._print("  /exit             ", "user", end="")
        print("- выход")
        print()
        self._print("💡 Можно писать команды на русском:", "info")
        print('  "прочитай проект ."')
        print('  "покажи файл bot.py"')
        print('  "прочитай файл src/main.py"')
        print()

    async def _handle_command(self, text: str) -> bool:
        """
        Обработать встроенную команду.

        Returns:
            True если команда обработана, False если это обычное сообщение
        """
        cmd = text.strip().lower()

        # /exit
        if cmd in ["/exit", "/quit", "exit", "quit", "выход"]:
            self._print("\n👋 До свидания!", "system")
            return True

        # /help
        if cmd in ["/help", "help", "помощь", "помоги"]:
            self._print_help()
            return True

        # /stats
        if cmd == "/stats":
            summary = self.token_tracker.format_summary()
            self._print(f"\n{summary}", "assistant")
            return True

        # Все остальные команды передаются агенту
        return False

    async def run(self) -> None:
        """Запустить консольное приложение."""
        self._print_banner()

        # Приветственное сообщение
        self._print("🤖 ", "assistant", end="")
        print("Привет! Я AI-агент для анализа кода.")
        print(f"   Я подключён к проекту: {self.project_path}")
        print("   Напиши 'прочитай проект .' чтобы начать анализ,")
        print("   или задай любой вопрос по коду.")
        print()

        while True:
            try:
                # Приглашение для ввода
                self._print("👤 Вы: ", "user", end="")
                user_input = input().strip()

                if not user_input:
                    continue

                # Проверяем встроенные команды
                if await self._handle_command(user_input):
                    if user_input.lower() in ["/exit", "/quit", "exit", "quit", "выход"]:
                        break
                    continue

                self.message_count += 1

                # Показываем что агент думает
                self._print("🤖 ", "assistant", end="")
                sys.stdout.flush()

                # Обрабатываем сообщение
                result = await self.agent.process_message(user_input)

                response = result["response"]
                duration = result.get("duration", 0)
                msg_type = result.get("type", "chat")

                # Выводим ответ
                print(response)

                # Выводим мета-информацию
                meta_parts = []

                if msg_type != "chat":
                    meta_parts.append(f"тип: {msg_type}")

                if duration > 0:
                    meta_parts.append(f"⏱️ {duration:.1f}с")

                if "token_usage" in result:
                    tu = result["token_usage"]
                    meta_parts.append(f"📥{tu['prompt_tokens']}+📤{tu['response_tokens']} токенов")

                if meta_parts:
                    self._print(f"   {' | '.join(meta_parts)}", "dim")

                print()

            except KeyboardInterrupt:
                print()
                self._print("\n👋 Прервано. До свидания!", "system")
                break
            except Exception as e:
                self._print(f"\n❌ Ошибка: {str(e)}", "error")
                print()


def main():
    """Точка входа."""
    import argparse

    parser = argparse.ArgumentParser(description="Консольный AI-агент для анализа кода")
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Модель Ollama (по умолчанию из .env или qwen2.5:0.5b)"
    )
    parser.add_argument(
        "-p", "--project",
        default=".",
        help="Путь к проекту для анализа (по умолчанию текущая директория)"
    )

    args = parser.parse_args()

    app = ConsoleApp(
        model_name=args.model,
        project_path=args.project,
    )

    asyncio.run(app.run())


if __name__ == "__main__":
    main()