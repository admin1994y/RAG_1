import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import Config
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """Строит FAISS индекс по чанкам."""
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dimension = embeddings.shape[1]
        # Нормализуем для косинусного сходства (inner product)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        logger.info("Индекс построен, размерность %d, кол-во векторов %d", dimension, len(chunks))
        return self

    def save(self, index_path=Config.INDEX_PATH, chunks_path=Config.CHUNKS_PATH):
        """Сохраняет индекс и чанки на диск."""
        if self.index is None:
            raise ValueError("Индекс не построен")
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Индекс сохранён в %s, чанки в %s", index_path, chunks_path)

    def load(self, index_path=Config.INDEX_PATH, chunks_path=Config.CHUNKS_PATH):
        """Загружает индекс и чанки с диска."""
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Файлы индекса не найдены")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info("Индекс загружен из %s, чанков %d", index_path, len(self.chunks))
        return self

    def search(self, query, k=10):
        """Ищет k ближайших чанков по запросу."""
        if self.index is None:
            raise ValueError("Индекс не загружен или не построен")
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, k)
        # Возвращаем список (chunk, score)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(scores[0][i])))
        return results