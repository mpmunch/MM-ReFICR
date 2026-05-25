import re
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Tuple

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def add_roles(context, trunc=4):
    role = ['Recommender','Seeker']
    for i in range(len(context)):
        context[i] = role[i%2] + ": " + context[i]

    context = ' '.join(context[-4:])
    return context

def search_number(text):
    match = re.search(r'\[(\d+)\]', text)

    if match:
        number = match.group(1)
        #print(number)
        return number
    else:
        return ""
    
def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()

def extract_movie_name(text):
    text = text.replace("-"," ")
    text = del_space(del_parentheses(text))
    #text = del_space(text)
    text = text.lower()
    return text


def recall_score(gt_list, pred_list, ks,verbose=True):
    hits = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            hits[k].append(len(list(set(gt).intersection(set(preds[:k]))))/len(gt))
    if verbose:
        for k in ks:
            print("Recall@{}: {:.4f}".format(k, np.mean(hits[k])))
    return hits

def _dcg_at_k(relevance: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = np.asarray(relevance[:k], dtype=np.float32)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2, dtype=np.float32))
    return float(np.sum(rel / discounts))


def ndcg_score(gt_list: List[List[int]], pred_list: List[List[int]], ks: List[int], verbose: bool = True) -> Dict[int, float]:
    ndcg_by_k: Dict[int, float] = {}
    for k in ks:
        sample_scores = []
        for gt, preds in zip(gt_list, pred_list):
            if len(gt) == 0:
                sample_scores.append(0.0)
                continue
            gt_set = set(gt)
            rel = [1 if p in gt_set else 0 for p in preds[:k]]
            dcg = _dcg_at_k(rel, k)
            ideal_rel = [1] * min(len(gt_set), k)
            idcg = _dcg_at_k(ideal_rel, k)
            sample_scores.append(0.0 if idcg == 0.0 else dcg / idcg)
        ndcg_by_k[k] = float(np.mean(sample_scores)) if sample_scores else 0.0
        if verbose:
            print(f"NDCG@{k}: {ndcg_by_k[k]:.4f}")
    return ndcg_by_k


def _extract_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu_score(pred_texts: List[str], ref_texts: List[str], max_n: int = 4, verbose: bool = True) -> float:
    clipped_counts = np.zeros(max_n, dtype=np.float64)
    total_counts = np.zeros(max_n, dtype=np.float64)
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(pred_texts, ref_texts):
        pred_tokens = pred.lower().strip().split()
        ref_tokens = ref.lower().strip().split()
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            pred_ngrams = _extract_ngrams(pred_tokens, n)
            ref_ngrams = _extract_ngrams(ref_tokens, n)
            total_counts[n - 1] += sum(pred_ngrams.values())
            for gram, cnt in pred_ngrams.items():
                clipped_counts[n - 1] += min(cnt, ref_ngrams.get(gram, 0))

    if pred_len == 0:
        if verbose:
            print("BLEU-4: 0.0000")
        return 0.0

    precisions = (clipped_counts + 1.0) / (total_counts + 1.0)
    geo_mean = float(np.exp(np.mean(np.log(precisions))))
    bp = 1.0 if pred_len > ref_len else float(np.exp(1.0 - (ref_len / max(pred_len, 1))))
    bleu = bp * geo_mean
    if verbose:
        print(f"BLEU-4: {bleu:.4f}")
    return bleu
def mrr_score(gt_list, pred_list, ks=None, verbose=True):
    if ks is None:
        ks = [1, 5, 10, 20, 50]

    mrr = {}
    for k in ks:
        reciprocal_ranks = []
        for gt, preds in zip(gt_list, pred_list):
            gt_set = set(gt)
            rr = 0.0
            for rank_idx, pred in enumerate(preds[:k], start=1):
                if pred in gt_set:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

        mrr[k] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    if verbose:
        for k in ks:
            print("MRR@{}: {:.4f}".format(k, mrr[k]))

    return mrr
