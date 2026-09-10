import logging
from transformers import pipeline
from config import Config

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.GENERATION_MODEL
        self.pipeline = None
        # Если используется локальная модель через transformers
        if "gpt" not in self.model_name.lower() and "yandex" not in self.model_name.lower():
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                logger.info("Локальная модель %s загружена", self.model_name)
            except Exception as e:
                logger.error("Не удалось загрузить локальную модель: %s", e)
                self.pipeline = None
        else:
            # Здесь можно реализовать вызов API (YandexGPT, GigaChat, OpenAI)
            logger.info("Используется API-модель (заглушка)")
            self.pipeline = None

    def generate(self, query, context_chunks):
        """
        Формирует ответ на основе запроса и списка чанков-контекста.
        """
        if not context_chunks:
            return "Не найдено релевантных инструкций для ответа."

        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""Ты – помощник, отвечающий на вопросы, используя только приведённые инструкции.
Не выдумывай информацию, не упоминай отсутствующие факты.

Вопрос: {query}

Инструкции:
{context}

Ответ:"""

        if self.pipeline is not None:
            # Локальная генерация
            try:
                response = self.pipeline(
                    prompt,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )[0]["generated_text"]
                # Извлекаем часть после "Ответ:"
                if "Ответ:" in response:
                    answer = response.split("Ответ:", 1)[1].strip()
                else:
                    answer = response.strip()
                return answer
            except Exception as e:
                logger.exception("Ошибка генерации: %s", e)
                return "Ошибка при генерации ответа."

        else:
            # Заглушка для API (можно заменить реальным вызовом)
            logger.warning("Генератор не настроен, возвращаем контекст")
            return f"Вот релевантные фрагменты инструкций:\n\n{context}"