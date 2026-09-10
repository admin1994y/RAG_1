import logging
import sys
from rag_system import RAGSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    print("Загрузка RAG-системы на основе файлов инструкций...")
    rag = RAGSystem(load_existing=True)
    print("Система готова. Введите 'exit' для выхода.")
    print("Для получения полной инструкции добавьте флаг -f, например: 'как авторизоваться? -f'")

    while True:
        user_input = input("\nВопрос: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break

        full_instr = False
        if user_input.endswith(" -f"):
            full_instr = True
            user_input = user_input[:-3].strip()

        if not user_input:
            continue

        answer = rag.answer(user_input, return_full_instruction=full_instr)
        print("\nОтвет:")
        print(answer)

if __name__ == "__main__":
    main()