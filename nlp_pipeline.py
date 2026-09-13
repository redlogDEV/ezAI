"""
NLP Pipeline: Tokenization → Embeddings → Softmax
Совместим с Python 3.8+ и работает на телефоне (Pydroid 3 / Termux)
Зависимости: только стандартная библиотека + numpy (pip install numpy)
"""

import json
import math
import random
import os
import sys

# ─────────────────────────────────────────────
# ПУТИ К ФАЙЛАМ
# ─────────────────────────────────────────────
INPUT_VOCAB_FILE   = "vocabulary.txt"       # входной словарь (одно слово на строку)
TOKENS_FILE        = "tokens.json"          # токенизированные слова
EMBEDDINGS_FILE    = "embeddings.json"      # эмбеддинги (ключ → вектор)
SOFTMAX_FILE       = "softmax_output.json"  # результат softmax

EMBEDDING_DIM = 16   # размерность вектора эмбеддинга
SEED = 42            # для воспроизводимости случайных эмбеддингов


# ─────────────────────────────────────────────
# ШАГИ ПАЙПЛАЙНА
# ─────────────────────────────────────────────

def detect_and_read(path: str) -> list[str]:
    """Автоматически определяет кодировку и читает файл."""
    encodings = ["utf-8-sig", "utf-8", "windows-1251", "cp866", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = [line.strip() for line in f if line.strip()]
            # Проверяем что прочитали нормально (не мусор)
            sample = " ".join(lines[:10])
            sample.encode("utf-8")
            print(f"  → Кодировка определена: {enc}")
            return lines
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    raise RuntimeError("Не удалось определить кодировку файла")


def load_vocabulary(path: str) -> list[str]:
    """Читает словарь из файла — одно слово/фраза на строку."""
    if not os.path.exists(path):
        # Ищем russian.txt рядом — конвертируем автоматически
        alt = os.path.join(os.path.dirname(path), "russian.txt")
        if os.path.exists(alt):
            print(f"[INFO] vocabulary.txt не найден, но найден {alt}")
            print("  → Конвертирую автоматически...")
            words = detect_and_read(alt)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(words))
            print(f"  → Сохранён как vocabulary.txt (UTF-8)")
            print(f"[1/4] Загружено слов: {len(words)}")
            return words

        print(f"[ERROR] Файл не найден: {path}")
        print("Создаю пример vocabulary.txt ...")
        example_words = [
            "привет", "мир", "python", "нейронная", "сеть",
            "токен", "эмбеддинг", "softmax", "модель", "обучение"
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(example_words))
        print(f"  → Создан пример: {path}")

    words = detect_and_read(path)
    print(f"[1/4] Загружено слов из словаря: {len(words)}")
    return words


def tokenize(words: list[str]) -> dict:
    """
    Простая токенизация:
      - каждому уникальному слову присваивается целочисленный токен-ID
      - результат: {слово: token_id, ...}
    """
    token_map = {}
    for idx, word in enumerate(words):
        clean = word.lower().strip()
        if clean and clean not in token_map:
            token_map[clean] = idx

    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(token_map, f, ensure_ascii=False, indent=2)

    print(f"[2/4] Токенизировано: {len(token_map)} уникальных токенов → {TOKENS_FILE}")
    return token_map


def generate_embeddings(token_map: dict) -> dict:
    """
    Генерирует случайные эмбеддинги для каждого токена.
    Ключ: слово (str), значение: вектор float длиной EMBEDDING_DIM.
    На практике это заменяется обученной таблицей весов (Embedding layer).
    """
    random.seed(SEED)

    def rand_vector(dim: int) -> list[float]:
        # инициализация Xavier/Glorot — стандарт для эмбеддингов
        limit = math.sqrt(6.0 / dim)
        return [round(random.uniform(-limit, limit), 6) for _ in range(dim)]

    embeddings = {}
    for word, token_id in token_map.items():
        embeddings[word] = {
            "token_id": token_id,
            "vector": rand_vector(EMBEDDING_DIM)
        }

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    print(f"[3/4] Эмбеддинги (dim={EMBEDDING_DIM}) сохранены → {EMBEDDINGS_FILE}")
    return embeddings


def softmax(vector: list[float]) -> list[float]:
    """Вычисляет softmax вектора (numerically stable)."""
    max_val = max(vector)
    exps = [math.exp(v - max_val) for v in vector]
    total = sum(exps)
    return [round(e / total, 8) for e in exps]


def apply_softmax(embeddings: dict) -> dict:
    """
    Применяет softmax к каждому вектору эмбеддинга.
    Результат: вероятностное распределение по размерностям вектора.
    """
    softmax_results = {}
    for word, data in embeddings.items():
        vec = data["vector"]
        sm  = softmax(vec)
        softmax_results[word] = {
            "token_id":       data["token_id"],
            "embedding":      vec,
            "softmax_output": sm,
            "top_dim":        sm.index(max(sm))   # индекс «горячей» размерности
        }

    with open(SOFTMAX_FILE, "w", encoding="utf-8") as f:
        json.dump(softmax_results, f, ensure_ascii=False, indent=2)

    print(f"[4/4] Softmax применён ко всем токенам → {SOFTMAX_FILE}")
    return softmax_results


def print_summary(softmax_results: dict, n: int = 5):
    """Печатает первые n результатов для быстрой проверки."""
    print("\n─── Пример результатов ─────────────────────────")
    for i, (word, data) in enumerate(softmax_results.items()):
        if i >= n:
            break
        sm_preview = [f"{v:.4f}" for v in data["softmax_output"][:4]]
        print(f"  '{word}' (id={data['token_id']}) → softmax[:4] = [{', '.join(sm_preview)}, ...]"
              f"  top_dim={data['top_dim']}")
    print("─────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  NLP Pipeline: токенизация → эмбеддинги → softmax")
    print("=" * 50 + "\n")

    # Шаг 1: загрузка словаря
    words = load_vocabulary(INPUT_VOCAB_FILE)

    # Шаг 2: токенизация
    token_map = tokenize(words)

    # Шаг 3: эмбеддинги
    embeddings = generate_embeddings(token_map)

    # Шаг 4: softmax
    softmax_results = apply_softmax(embeddings)

    # Итог
    print_summary(softmax_results)
    print("Все файлы сохранены в текущей папке:")
    for fname in [INPUT_VOCAB_FILE, TOKENS_FILE, EMBEDDINGS_FILE, SOFTMAX_FILE]:
        size = os.path.getsize(fname) if os.path.exists(fname) else 0
        print(f"  {fname:30s}  ({size} байт)")


if __name__ == "__main__":
    main()
