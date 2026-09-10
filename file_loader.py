import os
import re
from bs4 import BeautifulSoup
from config import Config
import logging

logger = logging.getLogger(__name__)

def clean_html(text):
    """Удаляет HTML-теги и преобразует сущности."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n")

def remove_guid_placeholders(text):
    """Удаляет маркеры вида {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}."""
    pattern = r'\{[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}'
    return re.sub(pattern, '', text)

def load_instructions_from_folder(folder_path=None):
    """
    Читает все .txt файлы из папки, извлекает текст, очищает его,
    возвращает список словарей: {'id': имя_файла, 'title': заголовок, 'content': полный_текст}
    """
    if folder_path is None:
        folder_path = Config.INSTRUCTIONS_DIR

    instructions = []
    if not os.path.exists(folder_path):
        logger.error("Папка %s не найдена", folder_path)
        return instructions

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            logger.warning("Не удалось прочитать %s: %s", filename, e)
            continue

        # Очистка
        clean_text = clean_html(raw_text)
        clean_text = remove_guid_placeholders(clean_text)
        # Удаляем лишние пустые строки
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            continue

        # Заголовок – берём первую строку, если она есть
        title = lines[0] if lines else filename.replace(".txt", "")

        instructions.append({
            "id": filename,
            "title": title,
            "content": clean_text
        })
        logger.info("Загружен файл: %s (символов: %d)", filename, len(clean_text))

    logger.info("Всего загружено %d инструкций", len(instructions))
    return instructions