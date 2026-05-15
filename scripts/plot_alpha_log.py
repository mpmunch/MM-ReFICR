#!/usr/bin/env python3

"""Plot alpha logging output.

This script consumes the JSONL produced by `alpha_log_path` during eval.
Each line should be a JSON object containing at least:
	- alpha: float
	- agreement: float

It writes two PNGs:
	1) Histogram of alpha
	2) Scatter of agreement (x) vs alpha (y), with correlation values
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_alpha_jsonl(path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
		"""Read alpha/agreement arrays from a JSONL file.

		Returns:
			alpha: [N] float64
			agreement: [N] float64
			rows: raw decoded JSON dicts (useful if you want to extend the script later)
		"""
	rows: List[Dict] = []
	alpha: List[float] = []
	agreement: List[float] = []

	with open(path, "r", encoding="utf-8") as f:
		for line_num, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
			except json.JSONDecodeError as e:
				raise ValueError(f"Invalid JSON on line {line_num} in {path}: {e}") from e

			if "alpha" not in rec or "agreement" not in rec:
				continue

			try:
				a = float(rec["alpha"])
				s = float(rec["agreement"])
			except (TypeError, ValueError):
				continue

			rows.append(rec)
			alpha.append(a)
			agreement.append(s)

	if len(alpha) == 0:
		raise ValueError(f"No valid rows found in {path} (expected keys: alpha, agreement)")

	return np.asarray(alpha, dtype=np.float64), np.asarray(agreement, dtype=np.float64), rows


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
	"""Pearson correlation (returns NaN for degenerate inputs)."""
	if x.size < 2 or y.size < 2:
		return float("nan")
	if np.allclose(x, x[0]) or np.allclose(y, y[0]):
		return float("nan")
	return float(np.corrcoef(x, y)[0, 1])


def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
	"""Assign ranks with average rank for ties.

	Returns 1-based ranks (standard for Spearman).
	"""
	x = np.asarray(x)
	n = x.size
	order = np.argsort(x, kind="mergesort")
	ranks = np.empty(n, dtype=np.float64)

	i = 0
	while i < n:
		j = i
		while j + 1 < n and x[order[j + 1]] == x[order[i]]:
			j += 1

		# average rank for indices i..j (inclusive), 1-based
		avg_rank = 0.5 * ((i + 1) + (j + 1))
		ranks[order[i : j + 1]] = avg_rank
		i = j + 1

	return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
	"""Spearman correlation computed as Pearson correlation of ranks."""
	if x.size < 2 or y.size < 2:
		return float("nan")
	rx = _rankdata_avg_ties(x)
	ry = _rankdata_avg_ties(y)
	return _pearson_corr(rx, ry)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot alpha/agreement logs (.jsonl) into PNG histogram and scatter plots."
	)
	parser.add_argument(
		"--input",
		required=True,
		help="Path to alpha log JSONL produced by alpha_log_path (must contain alpha + agreement).",
	)
	parser.add_argument(
		"--out_dir",
		default=None,
		help="Output directory for PNGs (defaults to the input file's directory).",
	)
	parser.add_argument("--bins", type=int, default=20, help="Number of bins for alpha histogram.")
	parser.add_argument(
		"--max_points",
		type=int,
		default=20000,
		help="Max points to plot in scatter (downsamples deterministically if larger).",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
	out_dir.mkdir(parents=True, exist_ok=True)

	alpha, agreement, _rows = _read_alpha_jsonl(str(input_path))
	n = int(alpha.size)

	# Correlations computed over the full dataset.
	pearson = _pearson_corr(alpha, agreement)
	spearman = _spearman_corr(alpha, agreement)

	# Scatter plots can get huge. To keep the PNG readable (and fast),
	# downsample deterministically if there are too many points.
	if args.max_points is not None and n > int(args.max_points):
		step = max(n // int(args.max_points), 1)
		idx = np.arange(0, n, step, dtype=np.int64)
		alpha_sc = alpha[idx]
		agreement_sc = agreement[idx]
	else:
		alpha_sc = alpha
		agreement_sc = agreement

	stem = input_path.stem
	hist_path = out_dir / f"{stem}_alpha_hist.png"
	scatter_path = out_dir / f"{stem}_alpha_vs_agreement.png"

	try:
		import matplotlib

		# Ensure this works on headless compute nodes.
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"matplotlib is required to save PNG plots. Install it (e.g. pip install matplotlib) "
			f"and re-run. Original import error: {e}"
		) from e

	# Histogram
	plt.figure(figsize=(7, 5))
	plt.hist(alpha, bins=int(args.bins), color="tab:blue", edgecolor="black", alpha=0.85)
	plt.title(f"Alpha histogram (n={n})")
	plt.xlabel("alpha")
	plt.ylabel("count")
	plt.tight_layout()
	plt.savefig(hist_path, dpi=200)
	plt.close()

	# Scatter
	plt.figure(figsize=(7, 5))
	plt.scatter(agreement_sc, alpha_sc, s=6, alpha=0.25)
	plt.title(f"Alpha vs agreement | Pearson={pearson:.4f} | Spearman={spearman:.4f}")
	plt.xlabel("agreement")
	plt.ylabel("alpha")
	plt.tight_layout()
	plt.savefig(scatter_path, dpi=200)
	plt.close()

	print(f"Read {n} rows from: {input_path}")
	print(f"Pearson(alpha, agreement):  {pearson}")
	print(f"Spearman(alpha, agreement): {spearman}")
	print(f"Wrote: {hist_path}")
	print(f"Wrote: {scatter_path}")


if __name__ == "__main__":
	main()