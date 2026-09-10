from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
import logging

logger = logging.getLogger(__name__)

def preprocess_instructions(instructions):
    """
    Принимает список словарей от file_loader, разбивает на чанки.
    Возвращает: [{'id': filename, 'chunk_id': int, 'text': str, 'metadata': {'title': title}}]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for instr in instructions:
        full_text = instr["content"]
        if not full_text:
            continue
        split_texts = splitter.split_text(full_text)
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "id": instr["id"],
                "chunk_id": i,
                "text": chunk,
                "metadata": {"title": instr["title"]}
            })
    logger.info("Создано %d чанков из %d инструкций", len(chunks), len(instructions))
    return chunks