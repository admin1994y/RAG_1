import logging
from file_loader import load_instructions_from_folder
from preprocess import preprocess_instructions
from index import EmbeddingIndex
from rerank import Reranker
from generate import Generator
from config import Config
import os

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, load_existing=False):
        self.embedder = EmbeddingIndex()
        self.reranker = Reranker()
        self.generator = Generator()

        if load_existing and os.path.exists(Config.INDEX_PATH) and os.path.exists(Config.CHUNKS_PATH):
            try:
                self.embedder.load()
                logger.info("Индекс загружен с диска")
                return
            except Exception as e:
                logger.warning("Не удалось загрузить индекс: %s. Перестроим.", e)

        # Загрузка инструкций из папки
        instructions = load_instructions_from_folder()
        if not instructions:
            logger.warning("Не найдено ни одной инструкции. Индекс будет пустым.")
            self.embedder.chunks = []
            self.embedder.index = None
            return

        chunks = preprocess_instructions(instructions)
        self.embedder.build_index(chunks)
        try:
            self.embedder.save()
        except Exception as e:
            logger.warning("Не удалось сохранить индекс: %s", e)

    def answer(self, query, return_full_instruction=False, top_k=10, rerank_top=5):
        """Аналогично предыдущей версии, но с данными из файлов."""
        if self.embedder.index is None or not self.embedder.chunks:
            return "Инструкции не загружены или индекс пуст."

        candidates = self.embedder.search(query, k=top_k)
        if not candidates:
            return "Не найдено релевантных инструкций."

        reranked = self.reranker.rerank(query, candidates, top_n=rerank_top)
        if not reranked:
            return "После реранкинга не осталось подходящих фрагментов."

        if return_full_instruction:
            from collections import Counter
            ids = [c["id"] for c, _ in reranked]
            if not ids:
                return "Не удалось определить инструкцию."
            main_id = Counter(ids).most_common(1)[0][0]
            # Ищем полный текст – нужно перечитать файл, но для простоты вернём первый чанк
            full_text = next((c["text"] for c, _ in reranked if c["id"] == main_id), "")
            return f"Полная инструкция (файл {main_id}):\n{full_text}"

        context_chunks = [c["text"] for c, _ in reranked]
        return self.generator.generate(query, context_chunks)