import requests
from config import Config
import logging

logger = logging.getLogger(__name__)

def fetch_instructions():
    """
    Получает список инструкций с API.
    Ожидается JSON вида: [{"id": ..., "title": ..., "description": ..., "steps": [...]}, ...]
    """
    payload = {"token": Config.API_TOKEN}
    try:
        response = requests.post(Config.API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logger.error("API вернул не список: %s", data)
            return []
        return data
    except Exception as e:
        logger.exception("Ошибка при получении инструкций: %s", e)
        return []