#!/usr/bin/env python3
"""ProbDPP HotpotQA experiments with heterogeneous source reliabilities.

This version intentionally removes homogeneous reliability experiments.
Every source has its own marginal reliability in every failure mode.

Default source reliabilities (mean approximately 0.70):
    [0.70,0.90,0.05,0.75,0.65,0.99,0.9,0.80,0.45,0.79]

Failure modes
-------------
independent:
    z_i ~ Bernoulli(alpha_i) independently.

correlated:
    Sources 1-5 share one Uniform(0,1) latent variable and sources 6-10
    share another. Within each group, z_i = 1{u_group < alpha_i}. This
    preserves every source marginal P(z_i=1)=alpha_i while introducing
    positive within-group correlation.

Methods
-------
    probdpp, kdpp, highest_alpha, random

The code performs exact search over all C(10,3)=120 subsets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests


DEFAULT_N = 10
DEFAULT_K = 3
DEFAULT_EPSILON = 0.6
DEFAULT_NUM_QUESTIONS = 1000
DEFAULT_NUM_SEEDS = 20
DEFAULT_GRAM_RIDGE = 1e-6
DEFAULT_ALPHA_VECTOR = (
    "0.70,0.90,0.05,0.75,0.65,0.99,0.9,0.80,0.45,0.79"
)

STATIONARY_METHODS = [
    "probdpp",
    "kdpp",
    "highest_alpha",
    "random",
]

ALL_METHODS = set(STATIONARY_METHODS)
ALL_FAILURE_MODES = {"independent", "correlated"}


# -----------------------------------------------------------------------------
# Utility and reproducibility
# -----------------------------------------------------------------------------


def stable_int_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_alpha_vector(value: str, n: int, name: str) -> np.ndarray:
    try:
        numbers = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated numeric vector") from exc
    if len(numbers) != n:
        raise ValueError(f"{name} must contain exactly N={n} values; got {len(numbers)}")
    vector = np.asarray(numbers, dtype=np.float64)
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"Every entry of {name} must lie in [0,1]")
    return vector


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def load_json_or_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {exc}"
                    ) from exc
                if isinstance(item, dict):
                    rows.append(item)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "examples", "rows"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
    raise ValueError("Unsupported dataset structure")


def ensure_gold_list(value: object) -> List[str]:
    if value is None:
        return [""]
    if isinstance(value, list):
        result = [str(x) for x in value if x is not None]
        return result if result else [""]
    return [str(value)]


def extract_question(example: Mapping[str, object]) -> str:
    for key in ("question", "input", "query"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Example has no question")


def extract_golds(example: Mapping[str, object]) -> List[str]:
    for key in ("answers", "answer", "gold_answers", "gold"):
        if key in example:
            return ensure_gold_list(example.get(key))
    return [""]


def format_context_item(item: object) -> Optional[str]:
    if isinstance(item, str):
        text = item.strip()
        return text or None

    if isinstance(item, dict):
        title = str(item.get("title", "")).strip()
        body = item.get("sentences", item.get("text", item.get("content", "")))
        if isinstance(body, list):
            body_text = " ".join(str(x).strip() for x in body if str(x).strip())
        else:
            body_text = str(body).strip()
        if not title and not body_text:
            return None
        return f"Title: {title}\n{body_text}".strip()

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        title = str(item[0]).strip()
        body = item[1]
        if isinstance(body, list):
            body_text = " ".join(str(x).strip() for x in body if str(x).strip())
        else:
            body_text = str(body).strip()
        if not title and not body_text:
            return None
        return f"Title: {title}\n{body_text}".strip()

    return None


def extract_candidate_passages(example: Mapping[str, object], n: int) -> List[str]:
    raw = None
    for key in (
        "candidate_passages",
        "passages",
        "contexts",
        "context",
        "retrieved_passages",
    ):
        if key in example:
            raw = example.get(key)
            break

    if not isinstance(raw, list):
        raise ValueError("No supported candidate passage list was found")

    passages: List[str] = []
    for item in raw:
        text = format_context_item(item)
        if text:
            passages.append(text)

    if len(passages) < n:
        raise ValueError(f"Expected at least {n} passages, found {len(passages)}")
    return passages[:n]


@dataclass(frozen=True)
class QAExample:
    example_id: str
    question: str
    golds: List[str]
    passages: List[str]


def prepare_examples(
    rows: Sequence[Mapping[str, object]],
    n: int,
    max_questions: int,
    shuffle_candidates: bool,
    candidate_seed: int,
) -> List[QAExample]:
    examples: List[QAExample] = []
    skipped = 0

    for row_index, row in enumerate(rows):
        if len(examples) >= max_questions:
            break
        try:
            question = extract_question(row)
            golds = extract_golds(row)
            passages = extract_candidate_passages(row, n)
        except ValueError as exc:
            skipped += 1
            print(f"[WARN] Skipping row {row_index}: {exc}")
            continue

        example_id = str(row.get("_id", row.get("id", row_index)))
        if shuffle_candidates:
            rng = np.random.default_rng(
                stable_int_seed("candidate_order", candidate_seed, example_id)
            )
            order = rng.permutation(n)
            passages = [passages[int(i)] for i in order]

        examples.append(
            QAExample(
                example_id=example_id,
                question=question,
                golds=golds,
                passages=passages,
            )
        )

    if not examples:
        raise RuntimeError("No valid examples were prepared")
    if len(examples) < max_questions:
        print(
            f"[WARN] Requested {max_questions} questions but prepared "
            f"{len(examples)}; skipped={skipped}"
        )
    return examples


# -----------------------------------------------------------------------------
# Passage embeddings and Gram matrices
# -----------------------------------------------------------------------------


class PassageEmbedder:
    def encode(self, passages: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(PassageEmbedder):
    def __init__(self, model_name: str, device: Optional[str], batch_size: int):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Install sentence-transformers or use --embedding-backend hashing"
            ) from exc

        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = int(batch_size)

    def encode(self, passages: Sequence[str]) -> np.ndarray:
        result = self.model.encode(
            list(passages),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float64)


class HashingEmbedder(PassageEmbedder):
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("Hash dimension must be positive")
        self.dim = int(dim)

    @staticmethod
    def hash64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def encode(self, passages: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(passages), self.dim), dtype=np.float64)
        for row_index, passage in enumerate(passages):
            tokens = re.findall(r"[a-z0-9]+", passage.lower())
            if not tokens:
                matrix[row_index, 0] = 1.0
                continue
            for token in tokens:
                hashed = self.hash64(token)
                column = hashed % self.dim
                sign = 1.0 if ((hashed >> 63) & 1) == 0 else -1.0
                matrix[row_index, column] += sign
            norm = float(np.linalg.norm(matrix[row_index]))
            if norm > 0:
                matrix[row_index] /= norm
            else:
                matrix[row_index, 0] = 1.0
        return matrix


def build_gram(embeddings: np.ndarray, ridge: float) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    gram = normalized @ normalized.T
    gram = 0.5 * (gram + gram.T)
    if ridge > 0:
        gram += float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    return gram.astype(np.float64)


# -----------------------------------------------------------------------------
# ProbDPP and DPP baselines
# -----------------------------------------------------------------------------


def reliability_reward(alpha: np.ndarray, epsilon: float) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    return 2.0 * (
        alpha * math.log(1.0 + epsilon)
        + (1.0 - alpha) * math.log(epsilon)
    )


def stable_logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        return -math.inf
    return float(value)


def precompute_subset_logdets(
    gram: np.ndarray,
    subsets: Sequence[Tuple[int, ...]],
) -> np.ndarray:
    values = np.empty(len(subsets), dtype=np.float64)
    for index, subset in enumerate(subsets):
        values[index] = stable_logdet(gram[np.ix_(subset, subset)])
    return values


def exact_best_subset(
    subsets: Sequence[Tuple[int, ...]],
    scores: np.ndarray,
) -> Tuple[int, ...]:
    if len(subsets) != len(scores):
        raise ValueError("Subset and score lengths differ")
    return subsets[int(np.argmax(scores))]


def select_subset(
    method: str,
    subsets: Sequence[Tuple[int, ...]],
    subset_logdets: np.ndarray,
    alpha_for_selection: Optional[np.ndarray],
    epsilon: float,
    random_rng: np.random.Generator,
    n: int,
    k: int,
) -> Tuple[int, ...]:
    if method == "probdpp":
        if alpha_for_selection is None:
            raise ValueError(f"{method} requires reliability values")
        weights = reliability_reward(alpha_for_selection, epsilon)
        scores = subset_logdets.copy()
        for index, subset in enumerate(subsets):
            scores[index] += float(weights[list(subset)].sum())
        return exact_best_subset(subsets, scores)

    if method == "kdpp":
        return exact_best_subset(subsets, subset_logdets)

    if method == "highest_alpha":
        if alpha_for_selection is None:
            raise ValueError(f"{method} requires reliability values")
        # Random jitter breaks exact reliability ties fairly and reproducibly.
        jitter = random_rng.uniform(0.0, 1e-12, size=n)
        ranking = sorted(
            range(n),
            key=lambda i: (-(float(alpha_for_selection[i]) + float(jitter[i])), i),
        )
        return tuple(sorted(ranking[:k]))

    if method == "random":
        selected = random_rng.choice(n, size=k, replace=False)
        return tuple(sorted(int(i) for i in selected))

    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# Reliability scenarios
# -----------------------------------------------------------------------------


def sample_failure_vector(
    failure_mode: str,
    alpha_t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(alpha_t)

    if failure_mode == "independent":
        return (rng.random(n) < alpha_t).astype(np.int64)

    if failure_mode == "correlated":
        if n % 2 != 0:
            raise ValueError("The two-group correlated model requires even N")
        split = n // 2
        u_group_1 = float(rng.random())
        u_group_2 = float(rng.random())
        z = np.empty(n, dtype=np.int64)
        z[:split] = (u_group_1 < alpha_t[:split]).astype(np.int64)
        z[split:] = (u_group_2 < alpha_t[split:]).astype(np.int64)
        return z

    raise ValueError(f"Unknown failure mode: {failure_mode}")


def alpha_used_by_method(
    method: str,
    failure_mode: str,
    alpha_t: np.ndarray,
) -> Optional[np.ndarray]:
    if method in {"kdpp", "random"}:
        return None

    if failure_mode in {"independent", "correlated"}:
        if method in {"probdpp", "highest_alpha"}:
            return alpha_t.copy()
        raise ValueError(f"Method {method} is not a supported method")

    raise ValueError(
        f"Unsupported method/failure-mode combination: {method}/{failure_mode}"
    )


# -----------------------------------------------------------------------------
# Ollama and persistent response cache
# -----------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are an expert question-answering system.\n"
    "Answer using ONLY the provided context.\n"
    "Rules:\n"
    "- Output exactly one line.\n"
    "- Use the format: FINAL: <answer>\n"
    "- Keep the answer as short as possible.\n"
    "- If the answer is not supported by the context, output: FINAL: Not stated.\n"
    "- Do not use outside knowledge.\n"
)


def build_prompt(context: str, question: str) -> str:
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "Context:\n" + (context or "").strip(),
            "Question:\n" + (question or "").strip(),
            "FINAL:",
        ]
    )


def parse_final(text: str) -> str:
    if not text:
        return "Not stated"
    for line in text.splitlines():
        if line.strip().lower().startswith("final:"):
            answer = line.split(":", 1)[1].strip()
            return answer if answer else "Not stated"
    first = text.splitlines()[0].strip()
    return first if first else "Not stated"


class SQLiteTextCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO cache(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        temperature: float,
        max_new_tokens: int,
        timeout_seconds: int,
        cache: SQLiteTextCache,
        retries: int,
    ):
        self.url = url
        self.model = model
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.cache = cache
        self.retries = int(retries)

    def generate(self, prompt: str) -> str:
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_new_tokens,
                    "prompt": prompt,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
            "stream": False,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                text = str(response.json().get("response", "")).strip()
                self.cache.set(cache_key, text)
                return text
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"Ollama request failed after {self.retries} attempts: {last_error}"
        )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_tokens(text: str) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match_single(prediction: str, gold: str) -> float:
    return 100.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1_single(prediction: str, gold: str) -> float:
    prediction_tokens = normalized_tokens(prediction)
    gold_tokens = normalized_tokens(gold)
    if not prediction_tokens and not gold_tokens:
        return 100.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 100.0 * (2.0 * precision * recall) / (precision + recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dynamic = [0] * (len(b) + 1)
    for token_a in a:
        previous = 0
        for index, token_b in enumerate(b, start=1):
            current = dynamic[index]
            if token_a == token_b:
                dynamic[index] = previous + 1
            else:
                dynamic[index] = max(dynamic[index], dynamic[index - 1])
            previous = current
    return dynamic[-1]


def rouge_l_single(prediction: str, gold: str) -> float:
    prediction_tokens = normalized_tokens(prediction)
    gold_tokens = normalized_tokens(gold)
    if not prediction_tokens and not gold_tokens:
        return 100.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(prediction_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(gold_tokens)
    return 100.0 * (2.0 * precision * recall) / (precision + recall)


def evaluate_prediction(prediction: str, golds: Sequence[str]) -> Dict[str, object]:
    golds = list(golds) if golds else [""]
    scores = []
    for gold in golds:
        scores.append(
            (
                token_f1_single(prediction, gold),
                exact_match_single(prediction, gold),
                rouge_l_single(prediction, gold),
                gold,
            )
        )
    best = max(scores, key=lambda item: (item[0], item[1], item[2]))
    return {
        "token_f1": max(item[0] for item in scores),
        "em": max(item[1] for item in scores),
        "rouge_l": best[2],
        "bert_ref": str(best[3]),
    }


def add_bertscore(
    rows: List[Dict[str, object]],
    enabled: bool,
    model_type: str,
    lang: str,
    rescale_with_baseline: bool,
    batch_size: int,
    device: Optional[str],
) -> None:
    if not enabled:
        for row in rows:
            row["bertscore"] = math.nan
        return

    try:
        from bert_score import score as bert_score
    except Exception as exc:
        print(f"[WARN] BERTScore unavailable: {exc}")
        for row in rows:
            row["bertscore"] = math.nan
        return

    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["bert_ref"]) for row in rows]

    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        kwargs = {
            "model_type": model_type,
            "lang": lang,
            "rescale_with_baseline": rescale_with_baseline,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        _, _, f1_values = bert_score(
            predictions[start:end],
            references[start:end],
            **kwargs,
        )
        values = f1_values.detach().cpu().numpy() * 100.0
        for offset, value in enumerate(values):
            rows[start + offset]["bertscore"] = float(value)
        print(f"[BERTScore] {end}/{len(rows)}")


# -----------------------------------------------------------------------------
# Output and aggregation
# -----------------------------------------------------------------------------


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else math.nan


def finite_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan
    if array.size == 1:
        return 0.0
    return float(np.std(array, ddof=1))


def aggregate_results(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metrics = ("token_f1", "rouge_l", "bertscore", "em")

    grouped_by_seed: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (str(row["failure_mode"]), str(row["method"]), int(row["seed"]))
        grouped_by_seed[key].append(row)

    seed_rows: List[Dict[str, object]] = []
    for (failure_mode, method, seed), group in sorted(grouped_by_seed.items()):
        output: Dict[str, object] = {
            "failure_mode": failure_mode,
            "method": method,
            "seed": seed,
            "n_questions": len(group),
        }
        for metric in metrics:
            output[metric] = finite_mean([float(row[metric]) for row in group])
        output["mean_selected_true_alpha"] = finite_mean(
            [float(row["mean_selected_true_alpha"]) for row in group]
        )
        output["mean_num_survivors"] = finite_mean(
            [float(row["num_surviving_selected"]) for row in group]
        )
        seed_rows.append(output)

    grouped_by_method: Dict[
        Tuple[str, str], List[Mapping[str, object]]
    ] = defaultdict(list)
    for row in seed_rows:
        grouped_by_method[(str(row["failure_mode"]), str(row["method"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (failure_mode, method), group in sorted(grouped_by_method.items()):
        output: Dict[str, object] = {
            "failure_mode": failure_mode,
            "method": method,
            "num_seeds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = finite_mean(values)
            output[f"{metric}_std"] = finite_std(values)
        for metric in ("mean_selected_true_alpha", "mean_num_survivors"):
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = finite_mean(values)
            output[f"{metric}_std"] = finite_std(values)
        summary_rows.append(output)

    return seed_rows, summary_rows


def print_summary(summary_rows: Sequence[Mapping[str, object]]) -> None:
    print("\n=== Mean +/- std across seeds ===")
    header = (
        f"{'Failure':<14} {'Method':<24} {'Token-F1':>16} "
        f"{'ROUGE-L':>16} {'BERTScore':>16} {'EM':>16} "
        f"{'SelAlpha':>12} {'Survivors':>12}"
    )
    print(header)
    print("-" * len(header))

    def format_metric(row: Mapping[str, object], name: str) -> str:
        mean = float(row[f"{name}_mean"])
        std = float(row[f"{name}_std"])
        if not np.isfinite(mean):
            return "NA"
        return f"{mean:.2f} +/- {std:.2f}"

    for row in summary_rows:
        print(
            f"{str(row['failure_mode']):<14} {str(row['method']):<24} "
            f"{format_metric(row, 'token_f1'):>16} "
            f"{format_metric(row, 'rouge_l'):>16} "
            f"{format_metric(row, 'bertscore'):>16} "
            f"{format_metric(row, 'em'):>16} "
            f"{format_metric(row, 'mean_selected_true_alpha'):>12} "
            f"{format_metric(row, 'mean_num_survivors'):>12}"
        )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def resolve_methods(failure_mode: str, requested: Optional[List[str]]) -> List[str]:
    if failure_mode not in ALL_FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {failure_mode}")
    methods = requested if requested else STATIONARY_METHODS
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return list(dict.fromkeys(methods))


def context_from_survivors(
    passages: Sequence[str],
    selected: Sequence[int],
    z: np.ndarray,
) -> str:
    surviving = [arm for arm in selected if int(z[arm]) == 1]
    blocks = []
    for rank, arm in enumerate(surviving, start=1):
        blocks.append(
            f"[Surviving passage {rank}; source {arm + 1}]\n{passages[arm]}"
        )
    return "\n\n".join(blocks)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_vector = parse_alpha_vector(args.alpha_vector, args.n, "--alpha-vector")

    print(f"[SCENARIO] alpha={alpha_vector.tolist()}")
    print(f"[SCENARIO] mean(alpha)={float(alpha_vector.mean()):.4f}")
    print(
        "[SCENARIO] correlated mode uses shared uniforms within groups 1-5 "
        "and 6-10, preserving each source marginal"
    )

    raw_rows = load_json_or_jsonl(Path(args.data))
    examples = prepare_examples(
        rows=raw_rows,
        n=args.n,
        max_questions=args.num_questions,
        shuffle_candidates=args.shuffle_candidates,
        candidate_seed=args.candidate_seed,
    )
    print(f"[INFO] Prepared {len(examples)} questions")

    if args.embedding_backend == "sentence_transformers":
        embedder: PassageEmbedder = SentenceTransformerEmbedder(
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    else:
        embedder = HashingEmbedder(args.hash_embedding_dim)

    cache = SQLiteTextCache(output_dir / "ollama_cache.sqlite3")
    llm = OllamaClient(
        url=args.ollama_url,
        model=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.ollama_timeout,
        cache=cache,
        retries=args.ollama_retries,
    )

    failure_modes = parse_csv_arg(args.failure_modes)
    invalid_modes = [mode for mode in failure_modes if mode not in ALL_FAILURE_MODES]
    if invalid_modes:
        raise ValueError(f"Unknown failure modes: {invalid_modes}")

    requested_methods = parse_csv_arg(args.methods) if args.methods else None
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    subsets = list(itertools.combinations(range(args.n), args.k))
    print(f"[INFO] Exact search evaluates {len(subsets)} subsets per method")

    subset_logdet_cache: List[np.ndarray] = []
    print("[RUN] Computing embeddings and Gram matrices")
    for index, example in enumerate(examples, start=1):
        embeddings = embedder.encode(example.passages)
        gram = build_gram(embeddings, args.gram_ridge)
        subset_logdet_cache.append(precompute_subset_logdets(gram, subsets))
        if index % 20 == 0 or index == len(examples):
            print(f"[EMBED] {index}/{len(examples)}")

    result_rows: List[Dict[str, object]] = []
    checkpoint_path = output_dir / "round_results_checkpoint.csv"

    try:
        for failure_mode in failure_modes:
            methods = resolve_methods(failure_mode, requested_methods)
            print(f"\n[MODE] {failure_mode}; methods={methods}")

            for seed in seeds:
                print(f"[SEED] {seed}")
                failure_rng = np.random.default_rng(
                    stable_int_seed("failure", failure_mode, seed)
                )

                for round_index, example in enumerate(examples, start=1):
                    alpha_t = alpha_vector.copy()

                    # One common failure realization for all methods in this round.
                    z = sample_failure_vector(
                        failure_mode=failure_mode,
                        alpha_t=alpha_t,
                        rng=failure_rng,
                    )

                    subset_logdets = subset_logdet_cache[round_index - 1]

                    for method in methods:
                        alpha_selection = alpha_used_by_method(
                            method=method,
                            failure_mode=failure_mode,
                            alpha_t=alpha_t,
                        )

                        method_rng = np.random.default_rng(
                            stable_int_seed(
                                "selection",
                                failure_mode,
                                seed,
                                round_index,
                                method,
                            )
                        )

                        selected = select_subset(
                            method=method,
                            subsets=subsets,
                            subset_logdets=subset_logdets,
                            alpha_for_selection=alpha_selection,
                            epsilon=args.epsilon,
                            random_rng=method_rng,
                            n=args.n,
                            k=args.k,
                        )

                        context = context_from_survivors(
                            example.passages, selected, z
                        )
                        prediction = parse_final(
                            llm.generate(build_prompt(context, example.question))
                        )
                        metrics = evaluate_prediction(prediction, example.golds)
                        surviving_selected = [
                            arm for arm in selected if int(z[arm]) == 1
                        ]

                        result_rows.append(
                            {
                                "failure_mode": failure_mode,
                                "method": method,
                                "seed": seed,
                                "round": round_index,
                                "example_id": example.example_id,
                                "question": example.question,
                                "golds": json.dumps(
                                    example.golds, ensure_ascii=False
                                ),
                                "prediction": prediction,
                                "bert_ref": metrics["bert_ref"],
                                "token_f1": float(metrics["token_f1"]),
                                "rouge_l": float(metrics["rouge_l"]),
                                "bertscore": math.nan,
                                "em": float(metrics["em"]),
                                "selected_indices": json.dumps(
                                    [arm + 1 for arm in selected]
                                ),
                                "surviving_selected_indices": json.dumps(
                                    [arm + 1 for arm in surviving_selected]
                                ),
                                "num_surviving_selected": len(surviving_selected),
                                "mean_selected_true_alpha": float(
                                    np.mean(alpha_t[list(selected)])
                                ),
                                "failure_vector": json.dumps(z.tolist()),
                                "true_alpha": json.dumps(alpha_t.tolist()),
                                "selection_alpha": (
                                    json.dumps(alpha_selection.tolist())
                                    if alpha_selection is not None
                                    else ""
                                ),
                            }
                        )


                    if round_index % args.checkpoint_every == 0:
                        write_csv(
                            checkpoint_path,
                            result_rows,
                            fieldnames=list(result_rows[0].keys()),
                        )
                        print(
                            f"[CHECKPOINT] mode={failure_mode} seed={seed} "
                            f"round={round_index}/{len(examples)} "
                            f"rows={len(result_rows)}"
                        )

        print("\n[RUN] Computing BERTScore")
        add_bertscore(
            rows=result_rows,
            enabled=args.compute_bertscore,
            model_type=args.bertscore_model,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale,
            batch_size=args.bertscore_batch_size,
            device=args.bertscore_device,
        )

        round_path = output_dir / "round_results.csv"
        write_csv(round_path, result_rows, list(result_rows[0].keys()))

        seed_rows, summary_rows = aggregate_results(result_rows)
        seed_path = output_dir / "seed_results.csv"
        summary_path = output_dir / "summary.csv"
        write_csv(seed_path, seed_rows, list(seed_rows[0].keys()))
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))

        scenario = {
            "n": args.n,
            "k": args.k,
            "epsilon": args.epsilon,
            "num_questions": len(examples),
            "num_seeds": args.num_seeds,
            "alpha_vector": alpha_vector.tolist(),
            "alpha_mean": float(alpha_vector.mean()),
            "correlated_groups": [list(range(1, 6)), list(range(6, 11))],
            "correlated_generator": (
                "For each group sample one shared Uniform(0,1); "
                "set z_i=1 if u_group < alpha_i"
            ),
            "failure_modes": failure_modes,
            "candidate_shuffle": bool(args.shuffle_candidates),
            "candidate_seed": args.candidate_seed,
        }
        with (output_dir / "scenario.json").open("w", encoding="utf-8") as handle:
            json.dump(scenario, handle, indent=2)

        print_summary(summary_rows)
        print(f"\n[OK] Round results: {round_path}")
        print(f"[OK] Seed results: {seed_path}")
        print(f"[OK] Summary: {summary_path}")
        print(f"[OK] Scenario: {output_dir / 'scenario.json'}")


    finally:
        cache.close()


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ProbDPP HotpotQA experiments with heterogeneous reliability "
            "under independent and correlated failures."
        )
    )
    parser.add_argument("--data", required=True, help="HotpotQA JSON or JSONL")
    parser.add_argument("--output-dir", default="probdpp_results_v2")

    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--alpha-vector",
        default=DEFAULT_ALPHA_VECTOR,
        help="Comma-separated source reliabilities for independent and correlated modes.",
    )
    parser.add_argument(
        "--failure-modes",
        default="independent,correlated",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method override",
    )

    parser.add_argument("--gram-ridge", type=float, default=DEFAULT_GRAM_RIDGE)

    parser.add_argument(
        "--embedding-backend",
        choices=["sentence_transformers", "hashing"],
        default="sentence_transformers",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--hash-embedding-dim", type=int, default=768)
    parser.add_argument(
        "--shuffle-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Randomly assign the ten HotpotQA passages to simulated source "
            "indices using a fixed per-question permutation."
        ),
    )
    parser.add_argument("--candidate-seed", type=int, default=12345)

    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--model-name", default="llama3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ollama-timeout", type=int, default=600)
    parser.add_argument("--ollama-retries", type=int, default=3)

    parser.add_argument(
        "--compute-bertscore",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument(
        "--bertscore-rescale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default=None)

    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n != 10:
        raise ValueError("This experiment is defined for N=10")
    if args.k <= 0 or args.k > args.n:
        raise ValueError("K must satisfy 1 <= K <= N")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.num_questions <= 0:
        raise ValueError("num-questions must be positive")
    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    parse_alpha_vector(args.alpha_vector, args.n, "--alpha-vector")

    modes = parse_csv_arg(args.failure_modes)
    if not modes:
        raise ValueError("At least one failure mode is required")
    unknown_modes = [mode for mode in modes if mode not in ALL_FAILURE_MODES]
    if unknown_modes:
        raise ValueError(f"Unknown failure modes: {unknown_modes}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
