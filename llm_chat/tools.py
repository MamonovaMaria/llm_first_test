import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Результат вызова инструмента."""
    tool_name: str
    success: bool
    data: Any = None
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def format_for_llm(self) -> str:
        """Форматировать результат для передачи в LLM."""
        if not self.success:
            return f"[Ошибка инструмента {self.tool_name}: {self.error}]"
        return f"[Результат {self.tool_name}]:\n{json.dumps(self.data, ensure_ascii=False, indent=2)}"


class ToolRegistry:
    """Реестр инструментов с trace-логированием."""

    def __init__(self, log_dir: str = "tool_logs"):
        self.tools: Dict[str, Any] = {}
        self.trace: list = []  # История вызовов
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def register(self, tool):
        """Зарегистрировать инструмент."""
        self.tools[tool.name] = tool
        print(f"🔧 Инструмент зарегистрирован: {tool.name} — {tool.description}")

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Вызвать инструмент и записать в trace.

        Args:
            tool_name: Имя инструмента
            **kwargs: Параметры для инструмента

        Returns:
            ToolResult с результатом
        """
        if tool_name not in self.tools:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Инструмент '{tool_name}' не найден. Доступны: {list(self.tools.keys())}"
            )
        else:
            try:
                tool = self.tools[tool_name]
                data = tool.execute(**kwargs)
                result = ToolResult(tool_name=tool_name, success=True, data=data)
            except Exception as e:
                result = ToolResult(tool_name=tool_name, success=False, error=str(e))

        # Записываем в trace
        self.trace.append(result.to_dict())
        self._write_trace(result)

        return result

    def _write_trace(self, result: ToolResult) -> None:
        """Записать trace в файл."""
        filename = f"trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.log_dir / filename

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

    def get_trace(self) -> list:
        """Получить историю вызовов."""
        return self.trace.copy()

    def format_trace(self) -> str:
        """Форматированный trace для вывода."""
        if not self.trace:
            return "📋 Trace: вызовов инструментов пока не было"

        lines = ["📋 *Trace вызовов инструментов:*\n"]
        for i, call in enumerate(self.trace, 1):
            status = "✅" if call["success"] else "❌"
            lines.append(f"{i}. {status} {call['tool']} ({call['timestamp'][:19]})")

        return "\n".join(lines)


# ========================================================================
# Инструменты
# ========================================================================

class FileCounter:
    """Подсчёт файлов в проекте."""

    name = "count_files"
    description = "Подсчитывает количество файлов в директории с фильтром по расширению"

    def execute(self, path: str = ".", extension: Optional[str] = None) -> Dict[str, Any]:
        target = Path(path).resolve()

        if not target.exists():
            raise FileNotFoundError(f"Путь не найден: {target}")
        if not target.is_dir():
            raise ValueError(f"Не директория: {target}")

        total_count = 0
        by_extension = {}

        # Используем os.scandir для быстрого обхода (быстрее чем os.walk)
        for root, dirs, files in os.walk(target):
            # Исключаем скрытые папки
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if file.startswith('.'):
                    continue

                _, ext = os.path.splitext(file)
                ext = ext or "(без расширения)"

                if extension is not None and ext != extension:
                    continue

                total_count += 1
                by_extension[ext] = by_extension.get(ext, 0) + 1

        return {
            "total_files": total_count,
            "path": str(target),
            "filter": extension or "все",
            "by_extension": by_extension,
            "files": [],
            "truncated": False,
        }


class WeatherTool:
    """Получение погоды через OpenWeatherMap API."""

    name = "get_weather"
    description = "Получает текущую погоду для города через OpenWeatherMap"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY", "")

    def execute(self, city: str) -> Dict[str, Any]:
        """
        Получить погоду.

        Args:
            city: Название города

        Returns:
            Словарь с погодой
        """
        if not self.api_key:
            # Демо-режим без API ключа
            return self._demo_weather(city)

        import requests

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "lang": "ru",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"]),
            "feels_like": round(data["main"]["feels_like"]),
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }

    def _demo_weather(self, city: str) -> Dict[str, Any]:
        """Демо-погода без API ключа (для тестов)."""
        import random

        conditions = ["ясно ☀️", "облачно ☁️", "дождь 🌧️", "снег ❄️", "гроза ⛈️"]

        return {
            "city": city,
            "country": "DEMO",
            "temperature": random.randint(-10, 35),
            "feels_like": random.randint(-15, 38),
            "description": random.choice(conditions),
            "humidity": random.randint(30, 90),
            "wind_speed": round(random.uniform(0, 15), 1),
            "note": "ДЕМО-РЕЖИМ. Добавьте OPENWEATHER_API_KEY в .env для реальных данных.",
        }


class WebSearchTool:
    """Поиск в интернете через DuckDuckGo (бесплатно, без API ключа)."""

    name = "web_search"
    description = "Ищет информацию в интернете через DuckDuckGo"

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Выполнить поиск.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Словарь с результатами
        """
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r["title"],
                        "url": r["href"],
                        "snippet": r["body"][:200],
                    })

            return {
                "query": query,
                "results_count": len(results),
                "results": results,
            }

        except ImportError:
            return {
                "query": query,
                "error": "Библиотека duckduckgo-search не установлена. Установите: pip install duckduckgo-search",
                "results": [],
            }


class DateTimeTool:
    """Инструмент для работы с датой и временем."""

    name = "datetime"
    description = "Возвращает текущую дату, время, день недели"

    def execute(self, timezone: str = "local") -> Dict[str, Any]:
        """
        Получить текущее время.

        Args:
            timezone: Часовой пояс (local или UTC)

        Returns:
            Словарь с датой и временем
        """
        now = datetime.now()

        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "day_of_week_ru": ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"][
                now.weekday()],
            "week_number": now.isocalendar()[1],
            "timezone": timezone,
        }


class CodeAnalyzer:
    """Анализ кода: подсчёт строк, функций, классов."""

    name = "analyze_code"
    description = "Анализирует Python-файл: считает строки, функции, классы, импорты"

    def execute(self, filepath: str) -> Dict[str, Any]:
        """
        Проанализировать Python-файл.

        Args:
            filepath: Путь к файлу

        Returns:
            Словарь с анализом
        """
        import re

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Подсчёт функций и классов
        functions = re.findall(r'^\s*def\s+(\w+)', content, re.MULTILINE)
        classes = re.findall(r'^\s*class\s+(\w+)', content, re.MULTILINE)
        imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(\S+)', content, re.MULTILINE)

        # Подсчёт строк кода (без пустых и комментариев)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        comment_lines = [l for l in lines if l.strip().startswith("#")]
        empty_lines = [l for l in lines if not l.strip()]

        return {
            "file": str(path),
            "total_lines": len(lines),
            "code_lines": len(code_lines),
            "comment_lines": len(comment_lines),
            "empty_lines": len(empty_lines),
            "functions": functions,
            "functions_count": len(functions),
            "classes": classes,
            "classes_count": len(classes),
            "imports": imports[:10],
            "imports_count": len(imports),
        }


class FileReadTool:
    """Чтение файлов и проектов (обёртка над ProjectReader)."""

    name = "read_file"
    description = "Читает файл или директорию проекта, находит специальные файлы"

    def __init__(self, project_reader):
        self.reader = project_reader

    def execute(self, path: str) -> Dict[str, Any]:
        """
        Прочитать файл или директорию.

        Args:
            path: Путь к файлу или директории

        Returns:
            Информация о файле/проекте
        """
        result = self.reader.read_project(path)
        return result
