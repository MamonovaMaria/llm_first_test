# src/llm_chat/agent.py
"""AI-агент с памятью, чтением файлов и системным промптом."""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from llm_chat.tools import (
    ToolRegistry, FileCounter, WeatherTool,
    WebSearchTool, DateTimeTool, CodeAnalyzer, FileReadTool
)


class AgentMemory:
    """Память агента — история диалога текущей сессии."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        """Добавить сообщение в историю."""
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        """Обрезать историю, если превышен лимит."""
        if len(self.messages) <= self.max_messages:
            return
        system_messages = [m for m in self.messages if m["role"] == "system"]
        other_messages = [m for m in self.messages if m["role"] != "system"]
        keep = self.max_messages - len(system_messages)
        self.messages = system_messages + other_messages[-keep:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Получить все сообщения."""
        return self.messages.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """Установить системный промпт (заменяет предыдущий)."""
        self.messages = [m for m in self.messages if m["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": prompt})

    def clear(self) -> None:
        """Очистить историю, сохранив системный промпт."""
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages
        self.created_at = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика памяти."""
        user_msgs = sum(1 for m in self.messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in self.messages if m["role"] == "assistant")
        system_msgs = sum(1 for m in self.messages if m["role"] == "system")
        total_chars = sum(len(m["content"]) for m in self.messages)

        return {
            "total_messages": len(self.messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "system_messages": system_msgs,
            "total_chars": total_chars,
            "session_age_minutes": round(
                (datetime.now() - self.created_at).total_seconds() / 60, 1
            ),
        }


class ProjectReader:
    """Чтение файлов проекта с распознаванием специальных файлов."""

    SPECIAL_FILES = [
        "AGENTS.md", "AGENT.md", "README.md",
        "CLAUDE.md", "CURSOR.md", "MY-AGENT.md",
        ".cursorrules", ".github/copilot-instructions.md",
    ]

    MAX_FILE_SIZE = 50_000
    MAX_FILES_IN_LIST = 100

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()

    def read_project(self, path: str = ".") -> Dict[str, Any]:
        """Прочитать файл или директорию."""
        target = self._resolve_path(path)

        if not target.exists():
            return {"error": f"Путь не найден: {target}"}

        result = {
            "path": str(target),
            "type": "file" if target.is_file() else "directory",
            "special_files": {},
            "structure": None,
        }

        if target.is_file():
            file_info = self._read_file(target)
            result.update(file_info)
        else:
            result["special_files"] = self._find_special_files(target)
            result["structure"] = self._get_directory_structure(target)

        return result

    def read_file(self, path: str) -> Dict[str, Any]:
        """Прочитать конкретный файл."""
        target = self._resolve_path(path)

        if not target.exists():
            return {"error": f"Файл не найден: {target}"}
        if target.is_dir():
            return {"error": f"Это директория, а не файл: {target}"}

        return self._read_file(target)

    def get_agent_instructions(self, directory: Path) -> Optional[str]:
        """Извлечь инструкции для агента из специальных файлов."""
        instruction_files = [
            "AGENTS.md", "AGENT.md", "CLAUDE.md", "CURSOR.md", "MY-AGENT.md",
        ]

        for filename in instruction_files:
            filepath = directory / filename
            if filepath.exists() and filepath.is_file():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    if content.strip():
                        return content
                except Exception:
                    pass

        readme_path = directory / "README.md"
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                first_lines = content[:200].lower()
                if any(kw in first_lines for kw in [
                    "ai agent", "instructions for ai", "агент", "инструкции для ии"
                ]):
                    return content
            except Exception:
                pass

        return None

    def _resolve_path(self, path: str) -> Path:
        """Разрешить путь относительно базового."""
        p = Path(path)
        return p if p.is_absolute() else self.base_path / p

    def _read_file(self, filepath: Path) -> Dict[str, Any]:
        """Прочитать файл с метаинформацией."""
        try:
            size = filepath.stat().st_size

            if size > self.MAX_FILE_SIZE:
                return {
                    "file_name": filepath.name,
                    "file_path": str(filepath),
                    "error": f"Файл слишком большой ({size:,} байт). Лимит: {self.MAX_FILE_SIZE:,}",
                    "size_bytes": size,
                }

            content = None
            for encoding in ["utf-8", "cp1251", "latin-1"]:
                try:
                    content = filepath.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return {
                    "file_name": filepath.name,
                    "file_path": str(filepath),
                    "error": "Не удалось прочитать файл (бинарный или неизвестная кодировка)",
                    "size_bytes": size,
                }

            return {
                "file_name": filepath.name,
                "file_path": str(filepath),
                "language": self._detect_language(filepath),
                "content": content,
                "lines": len(content.split("\n")),
                "size_bytes": size,
                "size_chars": len(content),
            }

        except Exception as e:
            return {
                "file_name": filepath.name,
                "file_path": str(filepath),
                "error": str(e),
            }

    def _find_special_files(self, directory: Path) -> Dict[str, str]:
        """Найти специальные файлы в директории."""
        found = {}

        for special_file in self.SPECIAL_FILES:
            filepath = directory / special_file
            if filepath.exists() and filepath.is_file():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    if len(content) > 5000:
                        content = content[:5000] + "\n\n... (файл обрезан)"
                    found[special_file] = content
                except Exception:
                    found[special_file] = "[бинарный файл или ошибка чтения]"

        return found

    def _get_directory_structure(self, directory: Path, max_depth: int = 3) -> str:
        """Получить структуру директории в виде дерева."""
        lines = [f"📁 {directory.name}/"]
        files_count = 0

        def walk(dir_path: Path, prefix: str = "", depth: int = 0):
            nonlocal files_count

            if depth > max_depth or files_count >= self.MAX_FILES_IN_LIST:
                if depth > max_depth:
                    lines.append(f"{prefix}...")
                return

            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except PermissionError:
                lines.append(f"{prefix}[доступ запрещён]")
                return

            for i, item in enumerate(items):
                if files_count >= self.MAX_FILES_IN_LIST:
                    break

                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")

                if item.name.startswith(".") and item.name not in [".cursorrules"]:
                    continue

                if item.is_dir():
                    lines.append(f"{prefix}{connector}📁 {item.name}/")
                    walk(item, new_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}📄 {item.name}")
                    files_count += 1

        walk(directory)

        if files_count >= self.MAX_FILES_IN_LIST:
            lines.append(f"\n... и ещё файлы (показано {self.MAX_FILES_IN_LIST})")

        return "\n".join(lines)

    def _detect_language(self, filepath: Path) -> str:
        """Определить язык программирования по расширению."""
        extension_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "React JSX", ".tsx": "React TSX", ".java": "Java",
            ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C",
            ".h": "C/C++ Header", ".cs": "C#", ".rb": "Ruby",
            ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
            ".scala": "Scala", ".md": "Markdown", ".json": "JSON",
            ".yaml": "YAML", ".yml": "YAML", ".xml": "XML",
            ".html": "HTML", ".css": "CSS", ".sql": "SQL",
            ".sh": "Shell", ".bash": "Bash", ".toml": "TOML",
            ".cfg": "Config", ".ini": "INI", ".env": "Environment",
            ".txt": "Text",
        }

        name = filepath.name.lower()
        if name == "dockerfile":
            return "Dockerfile"
        if name == "makefile":
            return "Makefile"

        return extension_map.get(filepath.suffix.lower(), f"Unknown ({filepath.suffix})")


class AIAgent:
    """AI-агент с памятью, чтением файлов и системным промптом."""

    ROLES = {
        "python_expert": """Ты — Senior Python-разработчик с 15-летним опытом.
Ты специализируешься на:
- Архитектуре высоконагруженных систем
- Асинхронном программировании (asyncio)
- Type hints и статическом анализе
- Оптимизации производительности
- Тестировании (pytest, unittest)

Твой стиль:
- Предлагаешь лучшие практики и паттерны
- Указываешь на антипаттерны
- Даёшь примеры кода с пояснениями
- Ссылаешься на PEP и документацию""",

        "code_reviewer": """Ты — эксперт по code review.
Ты анализируешь код и находишь:
- Баги и потенциальные ошибки
- Проблемы безопасности
- Узкие места производительности
- Нарушения принципов SOLID
- Отсутствие обработки ошибок
- Плохие названия переменных и функций

Формат ответа:
1. 🔴 Критические проблемы
2. 🟡 Предупреждения
3. 🟢 Рекомендации по улучшению
Для каждой проблемы — пример исправления.""",

        "teacher": """Ты — терпеливый преподаватель программирования.
Ты объясняешь концепции простыми словами, используешь аналогии.
Для каждой темы:
1. Объясняешь "на пальцах" зачем это нужно
2. Показываешь простой пример
3. Показываешь реальный пример из практики
4. Предупреждаешь о типичных ошибках""",

        "architect": """Ты — системный архитектор.
Ты проектируешь системы и объясняешь архитектурные решения.
Ты думаешь о:
- Масштабируемости
- Отказоустойчивости
- Связанности компонентов
- Паттернах проектирования
- Микросервисах vs монолите

Для каждого решения объясняешь trade-off'ы.""",
    }

    DEFAULT_SYSTEM_PROMPT = """Ты — AI-ассистент, который помогает разработчику с анализом кода.
Ты можешь:
- Читать файлы проекта
- Анализировать код и архитектуру
- Находить ошибки и предлагать исправления
- Отвечать на вопросы с учётом контекста диалога

При анализе кода:
- Обращай внимание на архитектуру и паттерны
- Указывай на потенциальные проблемы (безопасность, производительность)
- Предлагай конкретные улучшения с примерами кода
- Если не уверен — честно скажи об этом

Отвечай на русском языке, если пользователь пишет по-русски.
Код оформляй в markdown-блоках с указанием языка."""

    def __init__(self, llm_client, token_tracker=None):
        self.llm = llm_client
        self.token_tracker = token_tracker
        self.memory = AgentMemory()
        self.project_reader = ProjectReader()

        self.memory.set_system_prompt(self.DEFAULT_SYSTEM_PROMPT)

        self.tool_registry = ToolRegistry()
        self._register_tools()

        # Обновлённые команды — добавляем вызовы инструментов
        self.commands = {
            # Чтение проекта/файлов
            r"^(прочитай|прочитать|покажи|показать)\s+(файл|проект|директорию|папку)\s*(.*)": self._cmd_read_project,
            r"^(read|show)\s+(file|project|directory|dir)\s*(.*)": self._cmd_read_project,
            r"^/read\s*(.*)": self._cmd_read_project,
            r"^/file\s+(.+)": self._cmd_read_file,
            r"^(прочитай файл|покажи файл|read file|show file)\s+(.+)": self._cmd_read_file,

            # Инструменты
            r"^/count\s*(\.\w+)?$": self._cmd_count_files,
            r"^(сколько|посчитай|подсчитай)\s+(файлов|всех файлов|Python-файлов)\s*(.*)": self._cmd_count_files,
            r"^/weather\s+(.+)": self._cmd_weather,
            r"^(какая|какая сейчас)?\s*погода\s+(в|городе)\s+(.+)": self._cmd_weather,
            r"^/search\s+(.+)": self._cmd_search,
            r"^(найди|поищи|загугли)\s+(.+)": self._cmd_search,
            r"^/time$": self._cmd_datetime,
            r"^(который час|сколько времени|какое сегодня число|какой сегодня день)": self._cmd_datetime,
            r"^/analyze\s+(.+)": self._cmd_analyze,
            r"^/trace$": self._cmd_show_trace,

            # Системные
            r"^/clear$": self._cmd_clear,
            r"^/system\s+(.+)": self._cmd_set_system,
            r"^/system$": self._cmd_show_system,
            r"^/memory$": self._cmd_memory_stats,
            r"^/roles$": self._cmd_list_roles,
        }

    def _register_tools(self) -> None:
        """Зарегистрировать все инструменты."""

        self.tool_registry.register(FileCounter())
        self.tool_registry.register(WeatherTool())
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(DateTimeTool())
        self.tool_registry.register(CodeAnalyzer())
        self.tool_registry.register(FileReadTool(self.project_reader))

    async def _call_tool_and_respond(
            self, tool_name: str, tool_params: dict, user_message: str
    ) -> Dict[str, Any]:
        """
        Вызвать инструмент и передать результат модели для ответа.

        Обязательный вызов инструмента — модель не может ответить без него.
        """
        # 1. Вызываем инструмент
        result = self.tool_registry.call(tool_name, **tool_params)

        # 2. Добавляем в историю
        self.memory.add_message("user", user_message)
        self.memory.add_message(
            "system",
            f"[Вызван инструмент {tool_name}]:\n{result.format_for_llm()}"
        )

        # 3. Просим модель сформулировать ответ на основе результата
        prompt = (
            f"Пользователь спросил: '{user_message}'\n\n"
            f"Был вызван инструмент '{tool_name}'. Вот его результат:\n"
            f"{result.format_for_llm()}\n\n"
            f"Сформулируй понятный ответ на русском языке, используя ТОЛЬКО эти данные. "
            f"Не придумывай ничего от себя."
        )

        # 4. Отправляем модели
        import time
        start_time = time.time()

        def generate():
            return self.llm.chat_with_history(self.memory.get_messages())

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, generate)

        duration = time.time() - start_time
        self.memory.add_message("assistant", response)

        result_dict = {
            "response": response,
            "type": "tool_call",
            "duration": duration,
            "tool_result": result.to_dict(),
        }

        if self.token_tracker:
            usage = self.token_tracker.log(
                prompt=prompt,
                response=response,
                duration=duration,
            )
            result_dict["token_usage"] = usage

        return result_dict

        # ========================================================================
        # Обработчики команд с инструментами
        # ========================================================================

    async def _cmd_count_files(self, match: re.Match) -> Dict[str, Any]:
        ext = None
        user_text = match.group(0)

        if ".py" in user_text or "python" in user_text.lower():
            ext = ".py"

        return await self._call_tool_direct(
            tool_name="count_files",
            tool_params={"path": ".", "extension": ext},
            user_message=user_text,
            format_response=False,  # ← без модели
        )

    async def _cmd_weather(self, match: re.Match) -> Dict[str, Any]:
        city = match.groups()[-1].strip() if match.groups() else match.group(1).strip()

        return await self._call_tool_direct(
            tool_name="get_weather",
            tool_params={"city": city},
            user_message=f"погода в {city}",
            format_response=False,  # ← без модели
        )

    async def _cmd_search(self, match: re.Match) -> Dict[str, Any]:
        """Поиск через инструмент."""
        query = match.group(2) if match.lastindex >= 2 else match.group(1)
        query = query.strip()

        return await self._call_tool_and_respond(
            tool_name="web_search",
            tool_params={"query": query},
            user_message=f"найди {query}",
        )

    async def _cmd_datetime(self, match: re.Match) -> Dict[str, Any]:
        return await self._call_tool_direct(
            tool_name="datetime",
            tool_params={},
            user_message=match.group(0),
            format_response=False,  # ← без модели
        )

    async def _cmd_analyze(self, match: re.Match) -> Dict[str, Any]:
        """Анализ кода через инструмент."""
        path = match.group(1).strip()

        return await self._call_tool_and_respond(
            tool_name="analyze_code",
            tool_params={"filepath": path},
            user_message=f"проанализируй {path}",
        )

    async def _cmd_show_trace(self, match: re.Match) -> Dict[str, Any]:
        """Показать trace вызовов инструментов."""
        trace_text = self.tool_registry.format_trace()
        return {"response": trace_text, "type": "system"}

    async def _call_tool_direct(self, tool_name: str, tool_params: dict,
                                user_message: str, format_response: bool = True) -> Dict[str, Any]:
        """
        Вызвать инструмент и ответить.
        Если format_response=False — вернуть данные как есть, без модели.
        """
        import time

        # 1. Вызываем инструмент
        result = self.tool_registry.call(tool_name, **tool_params)

        if not result.success:
            return {"response": f"❌ {result.error}", "type": "error"}

        start_time = time.time()

        if format_response:
            # 2a. Форматируем через модель (для сложных ответов)
            self.memory.add_message("user", user_message)
            self.memory.add_message(
                "system",
                f"[Инструмент {tool_name}]:\n{result.format_for_llm()}"
            )

            prompt = (
                f"Пользователь: '{user_message}'\n"
                f"Результат инструмента:\n{result.format_for_llm()}\n"
                f"Напиши краткий ответ на русском."
            )

            def generate():
                return self.llm.chat_with_history(self.memory.get_messages())

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, generate)
            self.memory.add_message("assistant", response)
        else:
            # 2b. Без модели — просто показываем данные
            data = result.data
            response = self._format_tool_result(tool_name, data)
            duration = 0.0  # Мгновенно

        duration = time.time() - start_time if format_response else 0.0

        return {
            "response": response,
            "type": "tool_call",
            "duration": duration,
            "tool_result": result.to_dict(),
        }

    def _format_tool_result(self, tool_name: str, data: Dict[str, Any]) -> str:
        """Форматировать результат инструмента без модели."""

        if tool_name == "count_files":
            total = data.get("total_files", 0)
            by_ext = data.get("by_extension", {})
            path = data.get("path", "")

            lines = [f"📊 *Файлов в проекте ({path}): {total}*\n"]
            lines.append("По расширениям:")
            for ext, count in sorted(by_ext.items(), key=lambda x: -x[1]):
                lines.append(f"  • {ext}: {count}")

            if data.get("truncated"):
                lines.append("\n⚠️ _Показаны не все файлы_")

            return "\n".join(lines)

        if tool_name == "datetime":
            return (
                f"🕐 *{data.get('date')} {data.get('time')}*\n"
                f"📅 {data.get('day_of_week_ru')}, неделя {data.get('week_number')}"
            )

        if tool_name == "get_weather":
            return (
                f"🌤 *{data.get('city')}, {data.get('country')}*\n"
                f"🌡 {data.get('temperature')}°C (ощущается как {data.get('feels_like')}°C)\n"
                f"💧 Влажность: {data.get('humidity')}%\n"
                f"💨 Ветер: {data.get('wind_speed')} м/с\n"
                f"📝 {data.get('description', '')}"
            )

        # По умолчанию — JSON
        return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"

    # ========================================================================
    # Обработка сообщений
    # ========================================================================

    async def process_message(self, message: str, user_id: int = None) -> Dict[str, Any]:
        """Обработать сообщение пользователя."""
        for pattern, handler in self.commands.items():
            match = re.match(pattern, message, re.IGNORECASE)
            if match:
                print(f"🔍 Использую инструмент {handler.__name__}")
                return await handler(match)

        print(f"🔍 Генерирую ответ с помощью модели")
        return await self._chat(message)

    async def _chat(self, message: str) -> Dict[str, Any]:
        """Обычный диалог с моделью."""
        self.memory.add_message("user", message)
        messages = self.memory.get_messages()

        start_time = time.time()

        def generate():
            return self.llm.chat_with_history(messages)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, generate)

        duration = time.time() - start_time
        self.memory.add_message("assistant", response)

        result = {
            "response": response,
            "type": "chat",
            "duration": duration,
        }

        if self.token_tracker:
            usage = self.token_tracker.log(
                prompt=message,
                response=response,
                duration=duration,
            )
            result["token_usage"] = usage

        return result

    # ========================================================================
    # Команды
    # ========================================================================

    async def _cmd_read_project(self, match: re.Match) -> Dict[str, Any]:
        path = match.group(3).strip()
        if not path:
            path = "."

        result = self.tool_registry.call("read_file", path=path)

        if not result.success:
            return {"response": f"❌ {result.error}", "type": "error"}

        project_info = result.data

        if "error" in project_info:
            return {"response": f"❌ {project_info['error']}", "type": "error"}

        context = self._build_project_context(project_info)

        # === ВАЖНО: сохраняем содержимое файла в историю ===
        if project_info.get("type") == "file" or "content" in project_info:
            content = project_info.get("content", "")
            file_name = project_info.get("file_name", "")
            if isinstance(content, str) and content.strip():
                # Добавляем содержимое как системное сообщение — модель увидит его
                self.memory.add_message(
                    "system",
                    f"[Содержимое файла {file_name}]:\n```\n{content[:5000]}\n```"
                )

        # AGENTS.md
        target = self.project_reader._resolve_path(path)
        search_dir = target if target.is_dir() else target.parent
        instructions = self.project_reader.get_agent_instructions(search_dir)

        if instructions:
            combined = (
                f"{self.DEFAULT_SYSTEM_PROMPT}\n\n"
                f"--- ИНСТРУКЦИИ ИЗ ПРОЕКТА ---\n\n"
                f"{instructions}"
            )
            self.memory.set_system_prompt(combined)
            context += "\n\n📋 *Найдены инструкции для агента (применены как системный промпт)*"

        self.memory.add_message("user", f"[Прочитан: {path}]")
        self.memory.add_message("assistant", context)

        return {
            "response": context,
            "type": "tool_call",
            "tool_result": result.to_dict(),
        }

    async def _cmd_read_file(self, match: re.Match) -> Dict[str, Any]:
        """Обработка команды чтения файла."""
        if match.lastindex and match.lastindex >= 2:
            path = match.group(2).strip()
        else:
            path = match.group(1).strip()

        file_info = self.project_reader.read_file(path)

        if "error" in file_info:
            return {"response": f"❌ {file_info['error']}", "type": "error"}

        content = file_info.get("content", "")

        if not isinstance(content, str):
            return {
                "response": f"❌ Не удалось прочитать содержимое файла (тип: {type(content).__name__})",
                "type": "error",
            }

        language = file_info.get("language", "")
        file_name = file_info.get("file_name", "")
        file_path = file_info.get("file_path", "")
        lines = file_info.get("lines", 0)
        size_chars = len(content)

        truncated = False
        if len(content) > 3000:
            content = content[:3000]
            truncated = True

        lang_spec = language.lower() if language not in ("Unknown", "") else ""

        response = (
            f"📄 *{file_name}*\n"
            f"📁 `{file_path}`\n"
            f"📝 Язык: {language}\n"
            f"📏 Строк: {lines}\n\n"
            f"```{lang_spec}\n"
            f"{content}\n"
            f"```"
        )

        if truncated:
            response += f"\n\n⚠️ _Файл обрезан (показано 3000 из {size_chars} символов)_"

        self.memory.add_message("user", f"[Прочитан файл: {path}]")
        self.memory.add_message("assistant", response)

        return {
            "response": response,
            "type": "file_read",
            "file_info": file_info,
        }

    async def _cmd_clear(self, match: re.Match) -> Dict[str, Any]:
        """Очистка истории диалога."""
        self.memory.clear()
        return {"response": "✅ История диалога очищена. Новый чат начат.", "type": "system"}

    async def _cmd_set_system(self, match: re.Match) -> Dict[str, Any]:
        """Установка системного промпта (поддержка пресетов)."""
        text = match.group(1).strip()

        if text in self.ROLES:
            new_prompt = self.ROLES[text]
            preset_name = text
        else:
            new_prompt = text
            preset_name = None

        self.memory.set_system_prompt(new_prompt)

        if preset_name:
            response = f"✅ Роль установлена: *{preset_name}*\n\n{new_prompt[:300]}..."
        else:
            response = f"✅ Системный промпт обновлён:\n\n```\n{new_prompt[:500]}\n```"

        return {"response": response, "type": "system"}

    async def _cmd_show_system(self, match: re.Match) -> Dict[str, Any]:
        """Показать текущий системный промпт."""
        system_msgs = [m for m in self.memory.messages if m["role"] == "system"]

        if not system_msgs:
            return {"response": "❌ Системный промпт не задан", "type": "system"}

        prompt = system_msgs[-1]["content"]

        if len(prompt) > 1000:
            prompt = prompt[:1000] + "\n\n... (показано 1000 символов)"

        return {
            "response": f"⚙️ *Текущий системный промпт:*\n\n```\n{prompt}\n```",
            "type": "system",
        }

    async def _cmd_memory_stats(self, match: re.Match) -> Dict[str, Any]:
        """Показать статистику памяти."""
        stats = self.memory.get_stats()

        response = (
            "🧠 *Память агента*\n\n"
            f"📝 Сообщений: {stats['total_messages']}\n"
            f"   👤 Пользователь: {stats['user_messages']}\n"
            f"   🤖 Ассистент: {stats['assistant_messages']}\n"
            f"   ⚙️ Системных: {stats['system_messages']}\n"
            f"📏 Символов всего: {stats['total_chars']:,}\n"
            f"⏱️ Сессия: {stats['session_age_minutes']} мин"
        )

        return {"response": response, "type": "system"}

    async def _cmd_list_roles(self, match: re.Match) -> Dict[str, Any]:
        """Показать список доступных ролей."""
        roles = "\n".join(f"  • {name}" for name in self.ROLES.keys())

        return {
            "response": (
                "🎭 *Доступные роли:*\n\n"
                f"{roles}\n\n"
                "Использование: `/system python_expert`"
            ),
            "type": "system",
        }

    # ========================================================================
    # Вспомогательные методы
    # ========================================================================

    def _build_project_context(self, project_info: Dict[str, Any]) -> str:
        """Собрать контекст проекта для ответа."""
        parts = [f"📂 *Проект: {project_info['path']}*\n"]

        # Специальные файлы
        if project_info.get("special_files"):
            parts.append("📋 *Найдены специальные файлы:*\n")
            for filename, content in project_info["special_files"].items():
                if isinstance(content, str):
                    parts.append(f"\n**{filename}**:\n{content[:1000]}")
                    if len(content) > 1000:
                        parts.append("\n... (обрезано)")
                else:
                    parts.append(f"\n**{filename}**: [не удалось прочитать]")

        # Структура проекта
        structure = project_info.get("structure")
        if structure and isinstance(structure, str):
            parts.append(f"\n📁 *Структура проекта:*\n```\n{structure}\n```")

        # Одиночный файл
        content = project_info.get("content")
        if content and isinstance(content, str):
            file_name = project_info.get("file_name", "файла")
            parts.append(f"\n📄 *Содержимое {file_name}:*\n")
            parts.append(f"```\n{content[:3000]}\n```")

        parts.append("\n💡 _Теперь вы можете задавать вопросы по проекту._")

        return "\n".join(parts)