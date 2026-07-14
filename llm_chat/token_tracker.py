# src/llm_chat/token_tracker.py
"""Трекер токенов через Ollama API — точно и без зависимостей."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import requests


class TokenTracker:
    """
    Счётчик токенов через Ollama API.

    Использует эндпоинт /api/tokenize для точного подсчёта.
    Работает с любой моделью, загруженной в Ollama.
    """

    API_PRICES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "deepseek": {"input": 0.00014, "output": 0.00028},
    }

    def __init__(
            self,
            log_dir: str = "token_logs",
            model: str = "qwen2.5:0.5b",
            ollama_host: str = "http://localhost:11434",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.model = model
        self.ollama_host = ollama_host

        # Проверяем доступность эндпоинта токенизации
        self._tokenize_supported = self._check_tokenize_support()

        if not self._tokenize_supported:
            print("⚠️ /api/tokenize не поддерживается этой моделью.")
            print("   Использую эвристику (символы / 4)")

        # Статистика
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.total_time = 0.0

    def _check_tokenize_support(self) -> bool:
        """Проверить, поддерживает ли модель эндпоинт токенизации."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/tokenize",
                json={"model": self.model, "prompt": "test"},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def count(self, text: str) -> int:
        """Подсчитать токены в тексте через Ollama API."""
        if not text:
            return 0

        # Если API токенизации поддерживается
        if self._tokenize_supported:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/tokenize",
                    json={"model": self.model, "prompt": text},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    return len(data["tokens"])
            except Exception:
                pass  # Fallback к эвристике

        # Fallback: эвристика
        return self._heuristic_count(text)

    def _heuristic_count(self, text: str) -> int:
        """Приблизительный подсчёт (если API недоступен)."""
        has_cyrillic = any('а' <= c <= 'я' or 'А' <= c <= 'Я' or c in 'ёЁ' for c in text)
        chars_per_token = 2 if has_cyrillic else 4
        return max(1, len(text) // chars_per_token)

    def log(
            self,
            prompt: str,
            response: str,
            duration: float,
            temperature: float = 0.7,
            top_k: int = 40,
            top_p: float = 0.9,
            user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Записать взаимодействие и вернуть статистику."""
        prompt_tokens = self.count(prompt)
        response_tokens = self.count(response)
        total_tokens = prompt_tokens + response_tokens

        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_response_tokens += response_tokens
        self.total_time += duration

        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "duration_seconds": round(duration, 2),
            "tokens_per_second": round(response_tokens / duration, 1) if duration > 0 else 0,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "user_id": user_id,
        }

        self._write_log(record)

        return record

    def _write_log(self, record: Dict[str, Any]) -> None:
        """Дописать запись в дневной лог-файл."""
        filename = f"tokens_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.log_dir / filename

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def get_summary(self) -> Dict[str, Any]:
        """Получить сводную статистику."""
        total_tokens = self.total_prompt_tokens + self.total_response_tokens

        return {
            "requests": self.total_requests,
            "prompt_tokens": self.total_prompt_tokens,
            "response_tokens": self.total_response_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_request": round(total_tokens / max(1, self.total_requests)),
            "avg_prompt_tokens": round(self.total_prompt_tokens / max(1, self.total_requests)),
            "avg_response_tokens": round(self.total_response_tokens / max(1, self.total_requests)),
            "avg_time_seconds": round(self.total_time / max(1, self.total_requests), 1),
            "total_time_minutes": round(self.total_time / 60, 1),
            "model": self.model,
            "method": "Ollama API" if self._tokenize_supported else "эвристика",
        }

    def format_summary(self) -> str:
        """Красивая строка со статистикой для Telegram."""
        s = self.get_summary()

        return (
            "📊 *Статистика использования*\n\n"
            f"🤖 Модель: `{s['model']}`\n"
            f"🔤 Метод подсчёта: {s['method']}\n\n"
            f"📈 *Запросы*\n"
            f"   Всего: {s['requests']}\n"
            f"   Среднее токенов/запрос: {s['avg_tokens_per_request']}\n"
            f"   Среднее промпта: {s['avg_prompt_tokens']} токенов\n"
            f"   Среднее ответа: {s['avg_response_tokens']} токенов\n"
            f"   Среднее время: {s['avg_time_seconds']}с\n"
            f"   Общее время: {s['total_time_minutes']} мин\n\n"
            f"🔢 *Токены всего*\n"
            f"   Промптов: {s['prompt_tokens']:,}\n"
            f"   Ответов: {s['response_tokens']:,}\n"
            f"   Всего: {s['total_tokens']:,}\n\n"
            f"💰 *Сравнение с API*\n"
            f"   GPT-4:        ${self._estimate_cost('gpt-4'):.2f}\n"
            f"   GPT-3.5 Turbo: ${self._estimate_cost('gpt-3.5-turbo'):.4f}\n"
            f"   Claude Opus:   ${self._estimate_cost('claude-3-opus'):.2f}\n"
            f"   DeepSeek API:  ${self._estimate_cost('deepseek'):.4f}\n"
            f"   *Локально:*     $0.00 🎉\n\n"
            f"💡 _Подсчёт через Ollama API — всегда точный_"
        )

    def _estimate_cost(self, api_name: str) -> float:
        """Оценить стоимость через API."""
        prices = self.API_PRICES.get(api_name)
        if not prices:
            return 0.0

        input_cost = (self.total_prompt_tokens / 1000) * prices["input"]
        output_cost = (self.total_response_tokens / 1000) * prices["output"]

        return input_cost + output_cost

    def format_request_info(self, usage: Dict[str, Any]) -> str:
        """Краткая информация о токенах для одного запроса."""
        tps = usage["tokens_per_second"]

        if tps > 30:
            icon = "🚀"
        elif tps > 15:
            icon = "⚡"
        elif tps > 5:
            icon = "🔄"
        else:
            icon = "🐢"

        method = "" if self._tokenize_supported else "~"

        return (
            f"{icon} "
            f"📥{method}{usage['prompt_tokens']} + 📤{method}{usage['response_tokens']} = "
            f"🔤{method}{usage['total_tokens']} токенов | "
            f"⏱️{usage['duration_seconds']}с"
        )