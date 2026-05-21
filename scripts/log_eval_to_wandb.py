#!/usr/bin/env python3
import argparse
import re
import wandb
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
    parser.add_argument("--response_gen_file", required=True)  

    parser.add_argument("--step1_ok", required=True)
    parser.add_argument("--step2_ok", required=True)
    parser.add_argument("--step3_ok", required=True)
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
        "pipeline/all_steps_ok": int(
            to_int_bool(args.step1_ok)
            and to_int_bool(args.step2_ok)
            and to_int_bool(args.step3_ok)
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

    log_path = Path("logs") / f"{args.run_name}.log"
    if log_path.exists():
        wandb.save(str(log_path))

    response_gen_file = Path(args.response_gen_file)
    if response_gen_file.exists():   
        wandb.save(args.response_gen_file) 

    wandb.finish()

if __name__ == "__main__":
    main()
