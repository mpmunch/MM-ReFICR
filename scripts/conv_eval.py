#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import evaluate
import torch
import nltk
from nltk import ngrams as nltk_ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

SPECIAL_TOKENS = [
    "<pad>",
    "<|endoftext|>",
    "</s>",
    "<s>",
    "<bos>",
    "<eos>",
    "<unk>",
]

def clean_text(text: str) -> str:
    text = str(text or "")

    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("generation_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def read_jsonl(path: str, pred_key: str, ref_key: str) -> Tuple[List[str], List[str]]:
    predictions, references = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            pred = obj.get(pred_key, "") or ""
            ref = obj.get(ref_key, "") or ""

            predictions.append(clean_text(pred))
            references.append(clean_text(ref))

    return predictions, references


def compute_crs_bleu(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """
    CRS-style BLEU:
    - sentence-level NLTK BLEU
    - averaged over examples
    - bleu@1, bleu@2, bleu@3, bleu@4 use isolated n-gram weights
    """
    scores = {f"bleu@{k}": 0.0 for k in range(1, 5)}
    sent_cnt = 0
    smoothing = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        if not pred.strip():
            continue

        hyp = pred.split()
        refs = [ref.split()]

        if not hyp or not refs[0]:
            continue

        for k in range(1, 5):
            weights = [0.0, 0.0, 0.0, 0.0]
            weights[k - 1] = 1.0

            scores[f"bleu@{k}"] += sentence_bleu(
                refs,
                hyp,
                weights=tuple(weights),
                smoothing_function=smoothing,
            )

        sent_cnt += 1

    if sent_cnt == 0:
        return {**scores, "bleu_sent_cnt": 0}

    return {
        **{k: v / sent_cnt for k, v in scores.items()},
        "bleu_sent_cnt": sent_cnt,
    }

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
def calculate_distinct_n_standard(sentences, max_n=4):
    """
    Standard distinct-n metric for n in 1..max_n:
      - tokens = word_tokenize(sentence.lower())
      - distinct-n = unique_n / total_n for each n
    Returns a dict with keys "dist@1", ..., "dist@max_n"
    """
    results = {}
    for n in range(1, max_n + 1):
        total_ngrams = 0
        distinct_ngrams = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            if len(tokens) < n:
                continue
            ngs = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            distinct_ngrams.update(ngs)
            total_ngrams += len(ngs)
        if total_ngrams == 0:
            results[f"dist@{n}"] = 0.0
        else:
            results[f"dist@{n}"] = len(distinct_ngrams) / total_ngrams
    return results

def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict:
    rouge = evaluate.load("rouge")
    return rouge.compute(
        predictions=list(predictions),
        references=list(references),
        use_stemmer=True,
    )


def compute_perplexity(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int | None,
    device: str | None,
) -> dict:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(resolved_device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses = []
    token_counts = []

    clean_texts = [t if t.strip() else tokenizer.eos_token for t in texts]

    with torch.no_grad():
        for start in tqdm(range(0, len(clean_texts), batch_size), desc="PPL batches"):
            batch = clean_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
            ).to(resolved_device)

            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100

            outputs = model(**encoded, labels=labels)
            n_tokens = int((labels != -100).sum().item())

            losses.append(float(outputs.loss.item()) * n_tokens)
            token_counts.append(n_tokens)

    total_tokens = sum(token_counts)
    mean_nll = sum(losses) / total_tokens if total_tokens else float("inf")
    ppl = math.exp(mean_nll) if mean_nll < 100 else float("inf")

    return {
        "ppl_model": model_name,
        "perplexity": ppl,
        "mean_nll": mean_nll,
        "scored_tokens": total_tokens,
    }


def main() -> None:
    """
    run like this: 
    python scripts/conv_eval.py --input test_processed_gen1.jsonl --log-file metrics.log
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--log-file", default="generation_metrics.log")
    parser.add_argument("--pred-key", default="pred")
    parser.add_argument("--ref-key", default="resp")
    parser.add_argument("--ppl-model", default="gpt2")
    parser.add_argument("--ppl-batch-size", type=int, default=4)
    parser.add_argument("--ppl-max-length", type=int, default=1024)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    predictions, references = read_jsonl(args.input, args.pred_key, args.ref_key)

    if not predictions:
        raise ValueError("No examples found in input file.")

    logger.info("Input file: %s", Path(args.input).resolve())
    logger.info("Examples: %d", len(predictions))
    logger.info("Prediction key: %s | Reference key: %s", args.pred_key, args.ref_key)
    logger.info("Empty predictions: %d", sum(1 for p in predictions if not p))
    logger.info("Empty references: %d", sum(1 for r in references if not r))

    bleu_scores = compute_crs_bleu(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    dist_scores = calculate_distinct_unicrs(predictions, max_k=4) 
    dist_scores_standard = calculate_distinct_n_standard(predictions, 4) 

    logger.info("CRS-style BLEU: %s", json.dumps(bleu_scores, indent=2, sort_keys=True))
    logger.info("ROUGE: %s", json.dumps(rouge_scores, indent=2, sort_keys=True))
    logger.info("CRS-style Distinct: %s", json.dumps(dist_scores, indent=2, sort_keys=True))
    logger.info("Standard Distinct: %s", json.dumps(dist_scores_standard, indent=2, sort_keys=True))

    if not args.skip_ppl:
        ppl_scores = compute_perplexity(
            texts=predictions,
            model_name=args.ppl_model,
            batch_size=args.ppl_batch_size,
            max_length=args.ppl_max_length,
            device=args.device,
        )
        logger.info("Perplexity: %s", json.dumps(ppl_scores, indent=2, sort_keys=True))
    else:
        logger.info("Perplexity: skipped")

    logger.info("Done. Metrics written to %s", Path(args.log_file).resolve())


if __name__ == "__main__":
    main()
