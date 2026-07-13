import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class TokenTracker:
    """
    Простой счётчик и логгер токенов.

    Считает токены приблизительно (символы / 4 для английского,
    символы / 2 для русского) и сохраняет логи в JSONL.
    """

    def __init__(self, log_dir: str = "token_logs", model: str = "local_model"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.model = model

        # Простая статистика в памяти
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def count_tokens(self, text: str) -> int:
        """
        Приблизительный подсчёт токенов.

        Английский: ~4 символа на токен
        Русский: ~2 символа на токен (кириллица кодируется длиннее)
        """
        if not text:
            return 0

        # Определяем, есть ли кириллица
        has_cyrillic = any('а' <= c <= 'я' or 'А' <= c <= 'Я' or c in 'ёЁ' for c in text)

        if has_cyrillic:
            return max(1, len(text) // 2)
        else:
            return max(1, len(text) // 4)

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
        """
        Записать одно взаимодействие и вернуть статистику по нему.
        """
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(response)
        total_tokens = prompt_tokens + response_tokens

        # Обновляем статистику в памяти
        self.total_requests += 1
        self.total_tokens += total_tokens
        self.total_time += duration

        # Формируем запись
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

        # Сохраняем в файл
        self._save_to_file(record)

        return record

    def _save_to_file(self, record: Dict[str, Any]) -> None:
        """Сохранить запись в JSONL-файл (один файл в день)."""
        filename = f"tokens_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.log_dir / filename

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def get_summary(self) -> Dict[str, Any]:
        """Получить сводную статистику."""
        avg_tokens = self.total_tokens / max(1, self.total_requests)
        avg_time = self.total_time / max(1, self.total_requests)

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": round(avg_tokens, 1),
            "avg_time_seconds": round(avg_time, 2),
            "total_time_minutes": round(self.total_time / 60, 1),
            "model": self.model,
        }

    def format_summary(self) -> str:
        """Форматированная строка со статистикой."""
        s = self.get_summary()

        return (
            f"📊 *Статистика использования*\n\n"
            f"Модель: `{s['model']}`\n"
            f"Запросов: {s['total_requests']}\n"
            f"Токенов всего: {s['total_tokens']:,}\n"
            f"Среднее токенов/запрос: {s['avg_tokens_per_request']}\n"
            f"Среднее время ответа: {s['avg_time_seconds']} сек\n"
            f"Общее время генерации: {s['total_time_minutes']} мин\n\n"
            f"💰 *Экономия против API*\n"
            f"Локально: $0.00 (бесплатно)\n"
            f"GPT-4 API: ~${s['total_tokens'] / 1000 * 0.045:.4f}\n"
            f"Claude API: ~${s['total_tokens'] / 1000 * 0.045:.4f}"
        )