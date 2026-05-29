#!/usr/bin/env python3
import argparse
import csv
import os
import re
import wandb
from datetime import datetime
from pathlib import Path

RECALL_RE = re.compile(r"Recall@(\d+)\s*[:=]\s*([0-9]*\.?[0-9]+)")
NDCG_RE = re.compile(r"NDCG@(\d+)\s*[:=]\s*([0-9]*\.?[0-9]+)")
MRR_RE = re.compile(r"MRR@(\d+)\s*[:=]\s*([0-9]*\.?[0-9]+)")

def parse_metrics(path: str):
    metrics = {}
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return metrics

    text = p.read_text(encoding="utf-8", errors="ignore")
    for k, v in RECALL_RE.findall(text):
        metrics[f"recall@{k}"] = float(v)

    for k, v in NDCG_RE.findall(text):
        metrics[f"ndcg@{k}"] = float(v)  

    for k, v in MRR_RE.findall(text):
        metrics[f"MRR@{k}"] = float(v)  
    return metrics

def to_int_bool(x: str) -> int:
    return 1 if str(x).lower() in {"1", "true", "yes", "ok"} else 0

def append_csv_row(csv_path: str, run_name: str, args, payload: dict) -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "dataset": args.dataset,
        "model": args.model_path,
        "from_step": args.from_step,
        **{k: v for k, v in sorted(payload.items())},
    }
    existing_cols: list = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing_cols = next(csv.reader(f), [])
    all_cols = list(existing_cols) or list(row.keys())
    for k in row:
        if k not in all_cols:
            all_cols.append(k)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        if not existing_cols:
            writer.writeheader()
        writer.writerow(row)
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run_name", required=True)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--from_step", required=True)
    parser.add_argument("--model_path", default="")

    parser.add_argument("--conv2item_file", required=True)
    parser.add_argument("--ranking_file", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--response_gen_file", required=True)  

    parser.add_argument("--step1_ok", required=True)
    parser.add_argument("--step2_ok", required=True)
    parser.add_argument("--step3_ok", required=True)
    parser.add_argument("--step4_ok", required=True)
    args = parser.parse_args()
    conv2item = parse_metrics(args.conv2item_file)
    ranking = parse_metrics(args.ranking_file)

    print("conv2item parsed:", conv2item)
    print("ranking parsed:", ranking)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="evaluation",
        config={
            "dataset": args.dataset,
            "from_step": args.from_step,
            "model_path": args.model_path,
        },
    )

    payload = {
        "conv2item/status": to_int_bool(args.step1_ok),
        "conv2conv/status": to_int_bool(args.step2_ok),
        "ranking/status": to_int_bool(args.step3_ok),
        "response_gen/status": to_int_bool(args.step4_ok),
        "pipeline/all_steps_ok": int(
            to_int_bool(args.step1_ok)
            and to_int_bool(args.step2_ok)
            and to_int_bool(args.step3_ok)
            and to_int_bool(args.step4_ok)
        ),
    }

    for k, v in conv2item.items():
        payload[f"conv2item/{k}"] = v

    for k, v in ranking.items():
        payload[f"ranking/{k}"] = v
        payload[f"pipeline/final_{k}"] = v
        
    wandb.log(payload)
    for k, v in payload.items():
        run.summary[k] = v

    # CSV — one row per run, appended to logs/eval_results.csv
    log_path = Path(args.log_file)
    csv_path = str(log_path.parent / "eval_results.csv")
    row = append_csv_row(csv_path, args.run_name, args, payload)
    print(f"Metrics appended to: {csv_path}")

    # Wandb Table — single row for this run (avoids O(N²) storage across runs)
    table = wandb.Table(columns=list(row.keys()))
    table.add_data(*row.values())
    wandb.log({"eval_results": table})

    if log_path.exists():
        wandb.save(str(log_path))
    else:
        print(f"warning: eval log file not found, skipping wandb upload: {log_path}")

    response_gen_file = Path(args.response_gen_file)
    if to_int_bool(args.step4_ok) and response_gen_file.exists():
        wandb.save(str(response_gen_file))
    elif not to_int_bool(args.step4_ok):
        print("warning: step 4 did not complete — skipping response gen file upload")

    wandb.finish()

if __name__ == "__main__":
    main()
