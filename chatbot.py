"""
Чат-бот на основе датасета диалогов
Работает на телефоне (Termux / Pydroid 3)
Зависимости: только стандартная библиотека Python
"""

import os
import re
import math
import json

# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────
DATASET_FILES = ["dataset_part1.txt", "dataset_part2.txt"]  # файлы датасетов
HISTORY_FILE  = "chat_history.json"                         # история диалогов
TOP_K         = 3                                           # сколько похожих ответов искать


# ─────────────────────────────────────────────
# ЗАГРУЗКА ДАТАСЕТА
# ─────────────────────────────────────────────

def load_dataset(files: list) -> list[dict]:
    """Загружает датасет и разбивает на пары вопрос-ответ."""
    pairs = []
    for path in files:
        if not os.path.exists(path):
            print(f"  [!] Файл не найден: {path} — пропускаю")
            continue

        # Автоопределение кодировки
        for enc in ["utf-8-sig", "utf-8", "windows-1251", "cp866"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue

        # Разбиваем на блоки диалогов
        blocks = re.split(r"\n---+\n", text)
        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            i = 0
            while i < len(lines) - 1:
                if lines[i].startswith("USER:") and lines[i+1].startswith("ASSISTANT:"):
                    question = lines[i][5:].strip()
                    answer   = lines[i+1][10:].strip()
                    pairs.append({"q": question, "a": answer})
                i += 1

        print(f"  → {path}: загружено {len(pairs)} пар")

    return pairs


# ─────────────────────────────────────────────
# ПОИСК ПОХОЖЕГО ОТВЕТА (TF-IDF подобие)
# ─────────────────────────────────────────────

def tokenize_simple(text: str) -> list[str]:
    """Простая токенизация — только слова, нижний регистр."""
    return re.findall(r"[а-яёa-z]+", text.lower())


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Косинусное сходство двух TF векторов."""
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot    = sum(vec1[w] * vec2[w] for w in common)
    norm1  = math.sqrt(sum(v*v for v in vec1.values()))
    norm2  = math.sqrt(sum(v*v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def tf_vector(tokens: list[str]) -> dict:
    """Считает частоту каждого слова."""
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def find_best_answer(query: str, pairs: list[dict], top_k: int = 3) -> str:
    """Ищет наиболее похожий вопрос в датасете и возвращает ответ."""
    q_tokens = tokenize_simple(query)
    q_vec    = tf_vector(q_tokens)

    scores = []
    for pair in pairs:
        p_tokens = tokenize_simple(pair["q"])
        p_vec    = tf_vector(p_tokens)
        score    = cosine_similarity(q_vec, p_vec)
        scores.append((score, pair))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_pair = scores[0]

    # Если совсем нет похожего — честно говорим
    if best_score < 0.05:
        return "Извини, не уверен как ответить на это. Попробуй переформулировать?"

    return best_pair["a"]


# ─────────────────────────────────────────────
# ИСТОРИЯ ДИАЛОГОВ
# ─────────────────────────────────────────────

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Чат-бот на основе датасета диалогов")
    print("  Введи 'выход' чтобы завершить")
    print("  Введи 'история' чтобы посмотреть диалог")
    print("=" * 50 + "\n")

    print("Загружаю датасет...")
    pairs = load_dataset(DATASET_FILES)

    if not pairs:
        print("[ОШИБКА] Датасет пустой. Проверь наличие файлов dataset_part1.txt / dataset_part2.txt")
        return

    print(f"Загружено {len(pairs)} пар вопрос-ответ. Готов к диалогу!\n")

    history = load_history()

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["выход", "exit", "quit", "пока"]:
            print("Бот: Пока! Было приятно пообщаться!")
            break

        if user_input.lower() == "история":
            if not history:
                print("История пуста.\n")
            else:
                print("\n─── История диалога ───")
                for turn in history[-10:]:
                    print(f"Ты:  {turn['user']}")
                    print(f"Бот: {turn['bot']}\n")
            continue

        answer = find_best_answer(user_input, pairs)
        print(f"Бот: {answer}\n")

        history.append({"user": user_input, "bot": answer})
        save_history(history)


if __name__ == "__main__":
    main()
