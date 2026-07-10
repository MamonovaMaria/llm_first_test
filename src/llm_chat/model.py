"""Модуль для работы с Ollama API."""

import ollama
from typing import Optional, Dict, Any, List


class LLMClient:
    """Клиент для взаимодействия с локальной LLM через Ollama."""

    def __init__(self, model: str = "qwen2.5:0.5b", host: Optional[str] = None):
        self.model = model
        # Если нужно подключение к удаленному Ollama
        if host:
            import os
            os.environ["OLLAMA_HOST"] = host

    def list_models(self) -> List[Dict[str, Any]]:
        """Получить список доступных моделей."""
        try:
            return ollama.list().get("models", [])
        except Exception as e:
            raise ConnectionError(f"Не удалось подключиться к Ollama: {e}")

    def chat(
            self,
            prompt: str,
            temperature: float = 0.7,
            top_k: int = 40,
            top_p: float = 0.9,
            system_prompt: Optional[str] = None
    ) -> str:
        """Отправить запрос к модели."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )

        return response["message"]["content"]

    def compare_parameters(
            self,
            prompt: str,
            configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Сравнить ответы с разными параметрами."""
        results = []

        for config in configs:
            try:
                answer = self.chat(
                    prompt=prompt,
                    temperature=config.get("temperature", 0.7),
                    top_k=config.get("top_k", 40),
                    top_p=config.get("top_p", 0.9),
                )
                results.append({
                    "label": config.get("label", "Unknown"),
                    "params": {
                        "temperature": config.get("temperature", 0.7),
                        "top_k": config.get("top_k", 40),
                        "top_p": config.get("top_p", 0.9),
                    },
                    "answer": answer,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "label": config.get("label", "Unknown"),
                    "params": config.get("params", {}),
                    "answer": str(e),
                    "success": False
                })

        return results


# Предустановленные конфигурации для экспериментов
EXPERIMENT_CONFIGS = [
    {
        "label": "🎯 Точный",
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.5
    },
    {
        "label": "⚖️ Сбалансированный",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9
    },
    {
        "label": "🎨 Креативный",
        "temperature": 1.5,
        "top_k": 80,
        "top_p": 1.0
    }
]