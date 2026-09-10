import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Путь к папке с файлами инструкций
    INSTRUCTIONS_DIR = "instructions"  # можно задать абсолютный путь

    # Модели
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # или заменить на API

    # Параметры чанкинга
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Пути для сохранения индекса
    INDEX_PATH = "faiss_index.bin"
    CHUNKS_PATH = "chunks.pkl"