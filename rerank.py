from sentence_transformers import CrossEncoder
from config import Config
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.RERANKER_MODEL
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query, candidates, top_n=5):
        """
        candidates: список (chunk, score) от первичного поиска.
        Возвращает список (chunk, rerank_score) отсортированный по убыванию.
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c, _ in candidates]
        scores = self.model.predict(pairs)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]