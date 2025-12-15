import json
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams as nltk_ngrams

# Ensure you have the tokenizer data (matches your original file)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def calculate_distinct_n_standard(sentences, n):
    """
    Your original 'normal' metric:
      - tokens = word_tokenize(sentence.lower())
      - distinct-n = unique_n / total_n
    """
    total_ngrams = 0
    distinct_ngrams = set()

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        if len(tokens) < n:
            continue

        ngs = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        distinct_ngrams.update(ngs)
        total_ngrams += len(ngs)

    if total_ngrams == 0:
        return 0.0
    return len(distinct_ngrams) / total_ngrams

def calculate_distinct_unicrs(sentences, max_k=4):
    """
    UniCRS-style:
      - tokens = pred.split()
      - dist@k = len(unique k-grams across ALL preds) / sent_cnt
      - sent_cnt counts non-empty preds (after strip), even if too short for k-grams
    """
    uniq = {k: set() for k in range(1, max_k + 1)}
    sent_cnt = 0

    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue

        sent_cnt += 1
        tokens = s.split()

        for k in range(1, max_k + 1):
            for ng in nltk_ngrams(tokens, k):
                uniq[k].add(ng)

    report = {f"dist@{k}": (len(uniq[k]) / sent_cnt if sent_cnt else 0.0) for k in range(1, max_k + 1)}
    report["sent_cnt"] = sent_cnt
    return report

def clean_like_unicrs(s):
    # UniCRS removes these after decoding; harmless if they aren't present
    return (s or "").replace("<pad>", "").replace("<|endoftext|>", "").strip()

# Load your output file
predictions = []
output_file = "redial_test_processed_gen.jsonl"

try:
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "pred" in data:
                    predictions.append(clean_like_unicrs(data["pred"]))
            except json.JSONDecodeError:
                pass
except FileNotFoundError:
    print(f"Error: Could not find {output_file}. Make sure to run inference first.")
    raise SystemExit(1)

if not predictions:
    print("No predictions found in the file.")
else:
    # --- Standard (your original) ---
    std_d2 = calculate_distinct_n_standard(predictions, 2)
    std_d4 = calculate_distinct_n_standard(predictions, 4)

    # --- UniCRS style ---
    unicrs = calculate_distinct_unicrs(predictions, max_k=4)

    print("=== Standard Distinct-n (unique_n / total_n) ===")
    print(f"Dist-2: {std_d2}")
    print(f"Dist-4: {std_d4}")

    print("\n=== UniCRS dist@k (unique_kgrams / sent_cnt) ===")
    print(f"dist@1: {unicrs['dist@1']}")
    print(f"dist@2: {unicrs['dist@2']}")
    print(f"dist@3: {unicrs['dist@3']}")
    print(f"dist@4: {unicrs['dist@4']}")
    print(f"sent_cnt: {unicrs['sent_cnt']}")
