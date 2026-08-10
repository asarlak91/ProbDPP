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
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests


DEFAULT_N = 10
DEFAULT_K = 3
DEFAULT_EPSILON = 0.6
DEFAULT_NUM_QUESTIONS = 600
DEFAULT_NUM_SEEDS = 30
DEFAULT_CHANGE_POINT = 300
DEFAULT_BLOCK_SIZE = 10
DEFAULT_KL_UCB_C = 3.0
DEFAULT_GRAM_RIDGE = 1e-6
DEFAULT_ADAPTIVE_WINDOW = 100

# Controlled pre-change reliability vector, mean exactly 0.70.
# It is deliberately spread enough that interpolation toward its reversal can
# realize an extreme mean absolute shift of Delta_alpha=0.50 while preserving
# the same mean reliability.
DEFAULT_ALPHA_VECTOR = "1.0,1.0,1.0,1.0,0.9,0.7,0.5,0.4,0.3,0.2"
DEFAULT_SEVERITY_DELTAS = "0.05,0.15,0.30,0.50"
SEVERITY_NAMES = ["slight", "moderate", "large", "extreme"]

TIME_VARYING_METHODS = [
    "static_probdpp",
    "online_probdpp",
    "adaptive_probdpp",
    "oracle_probdpp",
]
ALL_METHODS = set(TIME_VARYING_METHODS)


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
# ProbDPP, DPP baselines, and semi-bandit learning
# -----------------------------------------------------------------------------


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps)
    q = min(max(float(q), eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_ucb_bernoulli(p_hat: float, pulls: int, t: int, c: float) -> float:
    if pulls <= 0 or t <= 1:
        return 1.0

    rhs = math.log(max(2, t))
    rhs += c * math.log(max(2.0, math.log(max(3, t))))

    low = min(max(float(p_hat), 0.0), 1.0)
    high = 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if pulls * kl_bernoulli(p_hat, middle) <= rhs:
            low = middle
        else:
            high = middle
    return low


def reliability_reward(alpha: np.ndarray, epsilon: float) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    return 2.0 * (
        alpha * math.log(1.0 + epsilon)
        + (1.0 - alpha) * math.log(epsilon)
    )


@dataclass
class SemiBanditState:
    n_arms: int
    window_size: int = 100
    pulls: np.ndarray = field(init=False)
    successes: np.ndarray = field(init=False)
    recent_outcomes: List[deque] = field(init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)
        self.successes = np.zeros(self.n_arms, dtype=np.int64)
        self.recent_outcomes = [
            deque(maxlen=self.window_size) for _ in range(self.n_arms)
        ]

    def empirical_means(self, prior: float) -> np.ndarray:
        """All-history empirical means used by the stationary online method."""
        result = np.full(self.n_arms, float(prior), dtype=np.float64)
        observed = self.pulls > 0
        result[observed] = self.successes[observed] / self.pulls[observed]
        return result

    def kl_ucb_indices(self, t: int, c: float) -> np.ndarray:
        """Standard all-history KL-UCB indices."""
        empirical = self.empirical_means(prior=0.0)
        result = np.ones(self.n_arms, dtype=np.float64)
        for arm in range(self.n_arms):
            result[arm] = kl_ucb_bernoulli(
                p_hat=float(empirical[arm]),
                pulls=int(self.pulls[arm]),
                t=int(t),
                c=float(c),
            )
        return result

    def sliding_window_kl_ucb_indices(self, t: int, c: float) -> np.ndarray:
        """KL-UCB based only on recent selected-source outcomes.

        This is an empirical nonstationary extension. Old feedback is forgotten
        once it leaves the per-source sliding window.
        """
        result = np.ones(self.n_arms, dtype=np.float64)
        effective_t = max(2, min(int(t), self.window_size))
        for arm, history in enumerate(self.recent_outcomes):
            pulls = len(history)
            if pulls == 0:
                result[arm] = 1.0
                continue
            p_hat = float(np.mean(np.asarray(history, dtype=np.float64)))
            result[arm] = kl_ucb_bernoulli(
                p_hat=p_hat,
                pulls=pulls,
                t=effective_t,
                c=float(c),
            )
        return result

    def update(self, selected: Sequence[int], z: np.ndarray) -> None:
        for arm in selected:
            outcome = int(z[arm])
            self.pulls[arm] += 1
            self.successes[arm] += outcome
            self.recent_outcomes[arm].append(outcome)


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
    alpha_for_selection: np.ndarray,
    epsilon: float,
    random_rng: np.random.Generator,
    n: int,
    k: int,
) -> Tuple[int, ...]:
    del random_rng, n, k  # kept in the signature for compatibility
    if method not in TIME_VARYING_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if alpha_for_selection is None:
        raise ValueError(f"{method} requires reliability values")

    weights = reliability_reward(alpha_for_selection, epsilon)
    scores = subset_logdets.copy()
    for index, subset in enumerate(subsets):
        scores[index] += float(weights[list(subset)].sum())
    return exact_best_subset(subsets, scores)


# -----------------------------------------------------------------------------
# Reliability scenarios
# -----------------------------------------------------------------------------


def true_alpha_for_round(
    failure_mode: str,
    t: int,
    alpha_first: np.ndarray,
    alpha_second: np.ndarray,
    change_point: int,
) -> np.ndarray:
    if failure_mode in {"independent", "correlated"}:
        return alpha_first.copy()
    if failure_mode == "time_varying":
        return alpha_first.copy() if t <= change_point else alpha_second.copy()
    raise ValueError(f"Unknown failure mode: {failure_mode}")


def sample_failure_vector(
    failure_mode: str,
    alpha_t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(alpha_t)

    if failure_mode in {"independent", "time_varying"}:
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
    alpha_t: np.ndarray,
    alpha_initial: np.ndarray,
    online_state: Optional[SemiBanditState],
    adaptive_state: Optional[SemiBanditState],
    t: int,
    kl_ucb_c: float,
    online_index: str,
    mean_prior: float,
) -> np.ndarray:
    if method == "static_probdpp":
        return alpha_initial.copy()

    if method == "oracle_probdpp":
        return alpha_t.copy()

    if method == "online_probdpp":
        if online_state is None:
            raise ValueError("online_probdpp requires online state")
        if online_index == "kl_ucb":
            return online_state.kl_ucb_indices(t=t, c=kl_ucb_c)
        if online_index == "mean":
            return online_state.empirical_means(prior=mean_prior)
        raise ValueError(f"Unknown online index: {online_index}")

    if method == "adaptive_probdpp":
        if adaptive_state is None:
            raise ValueError("adaptive_probdpp requires adaptive state")
        return adaptive_state.sliding_window_kl_ucb_indices(t=t, c=kl_ucb_c)

    raise ValueError(f"Unsupported time-varying method: {method}")


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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing moving average that ignores non-finite values."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()

    output = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            output[index] = float(np.mean(finite))
    return output


def cumulative_average(values: np.ndarray) -> np.ndarray:
    """Cumulative average that ignores non-finite values."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    running_sum = 0.0
    running_count = 0

    for index, value in enumerate(values):
        if np.isfinite(value):
            running_sum += float(value)
            running_count += 1
        if running_count > 0:
            output[index] = running_sum / running_count
    return output


def block_average(
    x: np.ndarray,
    values: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return non-overlapping block averages for a stable visual summary."""
    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if block_size <= 1:
        return x.copy(), values.copy()

    x_blocks: List[float] = []
    value_blocks: List[float] = []
    for start in range(0, len(values), block_size):
        end = min(len(values), start + block_size)
        x_chunk = x[start:end]
        value_chunk = values[start:end]
        finite = value_chunk[np.isfinite(value_chunk)]
        if len(x_chunk) == 0 or finite.size == 0:
            continue
        x_blocks.append(float(np.mean(x_chunk)))
        value_blocks.append(float(np.mean(finite)))

    return (
        np.asarray(x_blocks, dtype=np.float64),
        np.asarray(value_blocks, dtype=np.float64),
    )



# -----------------------------------------------------------------------------
# Four-severity time-varying experiment and integrated objective-gap plotting
# -----------------------------------------------------------------------------

DISPLAY_NAMES = {
    "static_probdpp": "Fixed ProbDPP (pre-change alpha)",
    "online_probdpp": "Online ProbDPP (all-history KL-UCB)",
    "adaptive_probdpp": "Adaptive ProbDPP (sliding-window KL-UCB)",
    "oracle_probdpp": "Oracle ProbDPP (gap = 0)",
}


def parse_float_csv(value: str, name: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated numeric list") from exc
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def mean_abs_change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def build_severity_vectors(
    alpha_before: np.ndarray,
    severity_deltas: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Generate after-change vectors with exact target mean absolute changes.

    We interpolate from alpha_before toward its reversal. This preserves the
    mean exactly and progressively changes which source indices are reliable.
    """
    if len(severity_deltas) != len(SEVERITY_NAMES):
        raise ValueError(
            f"Expected {len(SEVERITY_NAMES)} severity deltas, got {len(severity_deltas)}"
        )
    target = alpha_before[::-1].copy()
    max_delta = mean_abs_change(alpha_before, target)
    if max_delta <= 0:
        raise ValueError("alpha-vector must contain heterogeneous reliabilities")

    result: Dict[str, np.ndarray] = {}
    for name, requested_delta in zip(SEVERITY_NAMES, severity_deltas):
        if requested_delta < 0:
            raise ValueError("Severity deltas must be nonnegative")
        if requested_delta > max_delta + 1e-12:
            raise ValueError(
                f"Requested {name} Delta_alpha={requested_delta:.3f}, but reversing "
                f"the supplied alpha-vector only supports up to {max_delta:.3f}. "
                "Use the default alpha-vector or provide a more heterogeneous one."
            )
        lam = 0.0 if max_delta == 0 else requested_delta / max_delta
        after = (1.0 - lam) * alpha_before + lam * target
        after = np.clip(after, 0.0, 1.0)
        achieved = mean_abs_change(alpha_before, after)
        if abs(achieved - requested_delta) > 1e-9:
            raise RuntimeError(
                f"Could not realize requested Delta_alpha={requested_delta}; got {achieved}"
            )
        result[name] = after
    return result


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


def aggregate_severity_results(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    metrics = ("token_f1", "rouge_l", "bertscore", "em")

    grouped_seed: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_seed[(str(row["severity"]), str(row["method"]), int(row["seed"]))].append(row)

    seed_rows: List[Dict[str, object]] = []
    for (severity, method, seed), group in sorted(grouped_seed.items()):
        out: Dict[str, object] = {
            "severity": severity,
            "method": method,
            "seed": seed,
            "n_rounds": len(group),
        }
        for metric in metrics:
            out[metric] = finite_mean([float(r[metric]) for r in group])
        out["objective_gap"] = finite_mean([float(r["objective_gap"]) for r in group])
        out["mean_selected_true_alpha"] = finite_mean(
            [float(r["mean_selected_true_alpha"]) for r in group]
        )
        out["mean_num_survivors"] = finite_mean(
            [float(r["num_surviving_selected"]) for r in group]
        )
        seed_rows.append(out)

    grouped_summary: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped_summary[(str(row["severity"]), str(row["method"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (severity, method), group in sorted(grouped_summary.items()):
        out: Dict[str, object] = {
            "severity": severity,
            "method": method,
            "num_seeds": len(group),
        }
        for metric in (*metrics, "objective_gap", "mean_selected_true_alpha", "mean_num_survivors"):
            vals = [float(r[metric]) for r in group]
            out[f"{metric}_mean"] = finite_mean(vals)
            out[f"{metric}_std"] = finite_std(vals)
        summary_rows.append(out)

    # Pre/post summaries are useful for the paper table.
    phase_group: Dict[Tuple[str, str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        phase = "pre" if int(row["round"]) <= int(row["change_point"]) else "post"
        key = (str(row["severity"]), str(row["method"]), phase, int(row["seed"]))
        phase_group[key].append(row)

    phase_seed_rows: List[Dict[str, object]] = []
    for (severity, method, phase, seed), group in sorted(phase_group.items()):
        out: Dict[str, object] = {
            "severity": severity,
            "method": method,
            "phase": phase,
            "seed": seed,
            "n_rounds": len(group),
        }
        for metric in metrics:
            out[metric] = finite_mean([float(r[metric]) for r in group])
        out["objective_gap"] = finite_mean([float(r["objective_gap"]) for r in group])
        phase_seed_rows.append(out)

    phase_summary_group: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in phase_seed_rows:
        phase_summary_group[(str(row["severity"]), str(row["method"]), str(row["phase"]))].append(row)

    phase_summary_rows: List[Dict[str, object]] = []
    for (severity, method, phase), group in sorted(phase_summary_group.items()):
        out: Dict[str, object] = {
            "severity": severity,
            "method": method,
            "phase": phase,
            "num_seeds": len(group),
        }
        for metric in (*metrics, "objective_gap"):
            vals = [float(r[metric]) for r in group]
            out[f"{metric}_mean"] = finite_mean(vals)
            out[f"{metric}_std"] = finite_std(vals)
        phase_summary_rows.append(out)

    return seed_rows, summary_rows, phase_summary_rows


def make_objective_gap_blocks(
    rows: Sequence[Mapping[str, object]],
    block_size: int,
) -> List[Dict[str, object]]:
    """10-round block means within seed, then mean/95% CI across seeds."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # First average rounds within each seed/block.
    grouped: Dict[Tuple[str, str, int, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        block = (int(row["round"]) - 1) // block_size
        grouped[(str(row["severity"]), str(row["method"]), int(row["seed"]), block)].append(row)

    seed_blocks: List[Dict[str, object]] = []
    for (severity, method, seed, block), group in sorted(grouped.items()):
        seed_blocks.append(
            {
                "severity": severity,
                "method": method,
                "seed": seed,
                "block": block,
                "mean_gap": finite_mean([float(r["objective_gap"]) for r in group]),
                "first_round": min(int(r["round"]) for r in group),
                "last_round": max(int(r["round"]) for r in group),
            }
        )

    across: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_blocks:
        across[(str(row["severity"]), str(row["method"]), int(row["block"]))].append(row)

    summary: List[Dict[str, object]] = []
    for (severity, method, block), group in sorted(across.items()):
        vals = np.asarray([float(r["mean_gap"]) for r in group], dtype=np.float64)
        mean_gap = float(np.mean(vals))
        std_gap = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = std_gap / math.sqrt(max(1, len(vals)))
        first_round = min(int(r["first_round"]) for r in group)
        last_round = max(int(r["last_round"]) for r in group)
        summary.append(
            {
                "severity": severity,
                "method": method,
                "block": block,
                "mean_gap": mean_gap,
                "std_gap": std_gap,
                "num_seeds": len(vals),
                "ci95": 1.96 * sem,
                "first_round": first_round,
                "last_round": last_round,
                "x": 0.5 * (first_round + last_round),
            }
        )
    return summary


def plot_four_severity_panels(
    block_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    change_point: int,
    total_rounds: int,
    block_size: int,
    severity_deltas: Mapping[str, float],
    num_seeds: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True, sharey=True)
    axes = axes.ravel()
    plotted_methods = ["static_probdpp", "online_probdpp", "adaptive_probdpp"]

    # Global y maximum makes the four severities directly comparable.
    ymax = 0.0
    for r in block_rows:
        if str(r["method"]) in plotted_methods:
            ymax = max(ymax, float(r["mean_gap"]) + float(r["ci95"]))
    ymax = max(0.1, ymax * 1.08)

    for ax, severity in zip(axes, SEVERITY_NAMES):
        sev_rows = [r for r in block_rows if str(r["severity"]) == severity]
        for method in plotted_methods:
            mr = sorted(
                [r for r in sev_rows if str(r["method"]) == method],
                key=lambda r: int(r["block"]),
            )
            if not mr:
                continue
            x = np.asarray([float(r["x"]) for r in mr])
            y = np.asarray([float(r["mean_gap"]) for r in mr])
            ci = np.asarray([float(r["ci95"]) for r in mr])
            first = np.asarray([int(r["first_round"]) for r in mr])
            last = np.asarray([int(r["last_round"]) for r in mr])

            # Do not connect the last pre-change block to the first post-change block.
            pre_mask = last <= change_point
            post_mask = first > change_point
            line = None
            if np.any(pre_mask):
                line = ax.plot(
                    x[pre_mask], y[pre_mask], marker="o", linewidth=1.8,
                    label=DISPLAY_NAMES[method],
                )[0]
                ax.fill_between(
                    x[pre_mask],
                    np.maximum(0.0, y[pre_mask] - ci[pre_mask]),
                    y[pre_mask] + ci[pre_mask],
                    alpha=0.12,
                    color=line.get_color(),
                )
            if np.any(post_mask):
                if line is None:
                    line = ax.plot(
                        x[post_mask], y[post_mask], marker="o", linewidth=1.8,
                        label=DISPLAY_NAMES[method],
                    )[0]
                else:
                    ax.plot(
                        x[post_mask], y[post_mask], marker="o", linewidth=1.8,
                        color=line.get_color(),
                    )
                ax.fill_between(
                    x[post_mask],
                    np.maximum(0.0, y[post_mask] - ci[post_mask]),
                    y[post_mask] + ci[post_mask],
                    alpha=0.12,
                    color=line.get_color(),
                )

        ax.axhline(0.0, linestyle="--", linewidth=1.5, label="Oracle ProbDPP (gap = 0)")
        ax.axvline(float(change_point), linestyle=":", linewidth=1.5)
        ax.set_xlim(1, total_rounds)
        ax.set_ylim(-0.04, ymax)
        ax.set_title(
            f"{severity.capitalize()} change: $\\Delta_\\alpha$={severity_deltas[severity]:.2f}"
        )
        ax.grid(alpha=0.18)

    axes[2].set_xlabel("Round")
    axes[3].set_xlabel("Round")
    axes[0].set_ylabel("True ProbDPP objective gap\n(lower is better)")
    axes[2].set_ylabel("True ProbDPP objective gap\n(lower is better)")

    # One shared legend, deduplicated.
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(),
        loc="lower center", ncol=4, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"ProbDPP adaptation under four reliability-shift severities\n"
        f"T={total_rounds}, switch at t={change_point}, {block_size}-round blocks, "
        f"95% CI across {num_seeds} seeds",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Four-panel objective-gap figure: {output_path}")


def print_severity_summary(summary_rows: Sequence[Mapping[str, object]]) -> None:
    print("\n=== Mean +/- std across seeds ===")
    header = (
        f"{'Severity':<10} {'Method':<22} {'Token-F1':>16} {'ROUGE-L':>16} "
        f"{'BERTScore':>16} {'EM':>16} {'ObjGap':>14}"
    )
    print(header)
    print("-" * len(header))

    def fmt(row: Mapping[str, object], name: str) -> str:
        mean = float(row[f"{name}_mean"])
        std = float(row[f"{name}_std"])
        if not np.isfinite(mean):
            return "NA"
        return f"{mean:.2f} +/- {std:.2f}"

    for row in summary_rows:
        print(
            f"{str(row['severity']):<10} {str(row['method']):<22} "
            f"{fmt(row, 'token_f1'):>16} {fmt(row, 'rouge_l'):>16} "
            f"{fmt(row, 'bertscore'):>16} {fmt(row, 'em'):>16} "
            f"{fmt(row, 'objective_gap'):>14}"
        )


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_before = parse_alpha_vector(args.alpha_vector, args.n, "--alpha-vector")
    severity_values = parse_float_csv(args.severity_deltas, "--severity-deltas")
    after_vectors = build_severity_vectors(alpha_before, severity_values)
    severity_delta_map = {
        name: mean_abs_change(alpha_before, after_vectors[name]) for name in SEVERITY_NAMES
    }

    print(f"[SCENARIO] alpha_before={alpha_before.tolist()}")
    print(f"[SCENARIO] mean(alpha_before)={alpha_before.mean():.4f}")
    for name in SEVERITY_NAMES:
        after = after_vectors[name]
        print(
            f"[SCENARIO] {name}: Delta_alpha={severity_delta_map[name]:.4f}, "
            f"mean={after.mean():.4f}, alpha_after={after.tolist()}"
        )

    raw_rows = load_json_or_jsonl(Path(args.data))
    examples = prepare_examples(
        rows=raw_rows,
        n=args.n,
        max_questions=args.num_questions,
        shuffle_candidates=args.shuffle_candidates,
        candidate_seed=args.candidate_seed,
    )
    if len(examples) <= args.change_point:
        raise RuntimeError(
            f"Only {len(examples)} valid questions were prepared; change point "
            f"{args.change_point} requires more rounds."
        )
    print(f"[INFO] Prepared {len(examples)} questions/rounds")

    if args.embedding_backend == "sentence_transformers":
        embedder: PassageEmbedder = SentenceTransformerEmbedder(
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    else:
        embedder = HashingEmbedder(args.hash_embedding_dim)

    methods = parse_csv_arg(args.methods) if args.methods else TIME_VARYING_METHODS.copy()
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    required_for_plot = set(TIME_VARYING_METHODS)
    if not required_for_plot.issubset(set(methods)):
        missing = sorted(required_for_plot - set(methods))
        raise ValueError(f"The four-panel plot requires all four methods; missing: {missing}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    subsets = list(itertools.combinations(range(args.n), args.k))
    subset_to_index = {subset: i for i, subset in enumerate(subsets)}
    print(f"[INFO] Exact search evaluates {len(subsets)} subsets per method")

    # Embeddings/log-dets are independent of severity and seed, so compute once.
    subset_logdet_cache: List[np.ndarray] = []
    print("[RUN] Computing embeddings and Gram matrices once for all severities")
    for index, example in enumerate(examples, start=1):
        embeddings = embedder.encode(example.passages)
        gram = build_gram(embeddings, args.gram_ridge)
        subset_logdet_cache.append(precompute_subset_logdets(gram, subsets))
        if index % 20 == 0 or index == len(examples):
            print(f"[EMBED] {index}/{len(examples)}")

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

    result_rows: List[Dict[str, object]] = []
    checkpoint_path = output_dir / "round_results_checkpoint.csv"

    try:
        for severity in SEVERITY_NAMES:
            alpha_after = after_vectors[severity]
            delta_alpha = severity_delta_map[severity]
            print(f"\n[SEVERITY] {severity}; Delta_alpha={delta_alpha:.3f}")

            for seed in seeds:
                print(f"[SEED] severity={severity} seed={seed}")
                online_state = SemiBanditState(args.n, window_size=args.adaptive_window)
                adaptive_state = SemiBanditState(args.n, window_size=args.adaptive_window)

                for round_index, example in enumerate(examples, start=1):
                    alpha_t = (
                        alpha_before.copy()
                        if round_index <= args.change_point
                        else alpha_after.copy()
                    )

                    # Common random numbers across severities and methods:
                    # the same seed/round draws the same U_i; only alpha_t changes.
                    failure_rng = np.random.default_rng(
                        stable_int_seed("failure_uniform", seed, round_index)
                    )
                    failure_uniform = failure_rng.random(args.n)
                    z = (failure_uniform < alpha_t).astype(np.int64)

                    subset_logdets = subset_logdet_cache[round_index - 1]

                    # True objective over all subsets: diversity + current true reliability.
                    true_weights = reliability_reward(alpha_t, args.epsilon)
                    true_scores = subset_logdets.copy()
                    for subset_index, subset in enumerate(subsets):
                        true_scores[subset_index] += float(true_weights[list(subset)].sum())
                    oracle_best_objective = float(np.max(true_scores))

                    for method in methods:
                        alpha_selection = alpha_used_by_method(
                            method=method,
                            alpha_t=alpha_t,
                            alpha_initial=alpha_before,
                            online_state=online_state,
                            adaptive_state=adaptive_state,
                            t=round_index,
                            kl_ucb_c=args.kl_ucb_c,
                            online_index=args.online_index,
                            mean_prior=args.mean_prior,
                        )

                        method_rng = np.random.default_rng(
                            stable_int_seed("selection", severity, seed, round_index, method)
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

                        method_true_objective = float(true_scores[subset_to_index[selected]])
                        objective_gap = oracle_best_objective - method_true_objective
                        if objective_gap < 0 and objective_gap > -1e-9:
                            objective_gap = 0.0
                        if objective_gap < -1e-9:
                            raise RuntimeError(
                                f"Negative objective gap {objective_gap} at {severity}, "
                                f"seed={seed}, round={round_index}, method={method}"
                            )

                        context = context_from_survivors(example.passages, selected, z)
                        prediction = parse_final(
                            llm.generate(build_prompt(context, example.question))
                        )
                        metrics = evaluate_prediction(prediction, example.golds)
                        surviving_selected = [arm for arm in selected if int(z[arm]) == 1]

                        result_rows.append(
                            {
                                "failure_mode": "time_varying",
                                "severity": severity,
                                "delta_alpha": delta_alpha,
                                "method": method,
                                "seed": seed,
                                "round": round_index,
                                "change_point": args.change_point,
                                "phase": "pre" if round_index <= args.change_point else "post",
                                "example_id": example.example_id,
                                "question": example.question,
                                "golds": json.dumps(example.golds, ensure_ascii=False),
                                "prediction": prediction,
                                "bert_ref": metrics["bert_ref"],
                                "token_f1": float(metrics["token_f1"]),
                                "rouge_l": float(metrics["rouge_l"]),
                                "bertscore": math.nan,
                                "em": float(metrics["em"]),
                                "selected_indices": json.dumps([arm + 1 for arm in selected]),
                                "surviving_selected_indices": json.dumps(
                                    [arm + 1 for arm in surviving_selected]
                                ),
                                "num_surviving_selected": len(surviving_selected),
                                "mean_selected_true_alpha": float(
                                    np.mean(alpha_t[list(selected)])
                                ),
                                "failure_vector": json.dumps(z.tolist()),
                                "true_alpha": json.dumps(alpha_t.tolist()),
                                "selection_alpha": json.dumps(alpha_selection.tolist()),
                                "true_probdpp_objective": method_true_objective,
                                "oracle_best_objective": oracle_best_objective,
                                "objective_gap": objective_gap,
                            }
                        )

                        if method == "online_probdpp":
                            online_state.update(selected, z)
                        elif method == "adaptive_probdpp":
                            adaptive_state.update(selected, z)

                    if round_index % args.checkpoint_every == 0:
                        write_csv(
                            checkpoint_path,
                            result_rows,
                            fieldnames=list(result_rows[0].keys()),
                        )
                        print(
                            f"[CHECKPOINT] severity={severity} seed={seed} "
                            f"round={round_index}/{len(examples)} rows={len(result_rows)}"
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

        seed_rows, summary_rows, phase_summary_rows = aggregate_severity_results(result_rows)
        seed_path = output_dir / "seed_results.csv"
        summary_path = output_dir / "summary.csv"
        phase_path = output_dir / "phase_summary.csv"
        write_csv(seed_path, seed_rows, list(seed_rows[0].keys()))
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
        write_csv(phase_path, phase_summary_rows, list(phase_summary_rows[0].keys()))

        block_rows = make_objective_gap_blocks(result_rows, args.block_size)
        block_path = output_dir / f"objective_gap_{args.block_size}round_blocks.csv"
        write_csv(block_path, block_rows, list(block_rows[0].keys()))

        scenario = {
            "n": args.n,
            "k": args.k,
            "epsilon": args.epsilon,
            "num_questions": len(examples),
            "num_seeds": args.num_seeds,
            "change_point": args.change_point,
            "block_size": args.block_size,
            "adaptive_window": args.adaptive_window,
            "alpha_before": alpha_before.tolist(),
            "alpha_before_mean": float(alpha_before.mean()),
            "severity_deltas": severity_delta_map,
            "alpha_after": {name: after_vectors[name].tolist() for name in SEVERITY_NAMES},
            "alpha_after_means": {
                name: float(after_vectors[name].mean()) for name in SEVERITY_NAMES
            },
            "severity_construction": (
                "alpha_after=(1-lambda)*alpha_before + lambda*reverse(alpha_before), "
                "with lambda selected to achieve the requested mean absolute Delta_alpha; "
                "this preserves mean reliability."
            ),
            "methods": methods,
            "candidate_shuffle": bool(args.shuffle_candidates),
            "candidate_seed": args.candidate_seed,
            "online_note": "Online ProbDPP uses all-history KL-UCB.",
            "adaptive_note": (
                "Adaptive ProbDPP uses sliding-window KL-UCB as an empirical "
                "nonstationary extension; it is not covered by the stationary regret theorem."
            ),
        }
        scenario_path = output_dir / "scenario.json"
        with scenario_path.open("w", encoding="utf-8") as handle:
            json.dump(scenario, handle, indent=2)

        fig_path = output_dir / "time_varying_four_severity_objective_gap.png"
        plot_four_severity_panels(
            block_rows=block_rows,
            output_path=fig_path,
            change_point=args.change_point,
            total_rounds=len(examples),
            block_size=args.block_size,
            severity_deltas=severity_delta_map,
            num_seeds=args.num_seeds,
        )

        # Sanity check: Oracle gap must be zero in every severity.
        oracle_max = max(
            float(r["objective_gap"])
            for r in result_rows
            if str(r["method"]) == "oracle_probdpp"
        )
        print(f"[CHECK] Oracle max objective gap={oracle_max:.3e} (should be 0)")
        print_severity_summary(summary_rows)

        print(f"\n[OK] Round results: {round_path}")
        print(f"[OK] Seed results: {seed_path}")
        print(f"[OK] Overall summary: {summary_path}")
        print(f"[OK] Pre/post summary: {phase_path}")
        print(f"[OK] Objective-gap blocks: {block_path}")
        print(f"[OK] Scenario: {scenario_path}")
        print(f"[OK] Four-panel figure: {fig_path}")

    finally:
        cache.close()


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the four-severity ProbDPP time-varying HotpotQA experiment and "
            "save downstream metrics plus a 2x2 true-objective-gap figure."
        )
    )
    parser.add_argument("--data", required=True, help="HotpotQA JSON or JSONL")
    parser.add_argument("--output-dir", default="probdpp_four_severity_results")

    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--change-point", type=int, default=DEFAULT_CHANGE_POINT)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--alpha-vector",
        default=DEFAULT_ALPHA_VECTOR,
        help="Pre-change reliability vector. Default mean is exactly 0.70.",
    )
    parser.add_argument(
        "--severity-deltas",
        default=DEFAULT_SEVERITY_DELTAS,
        help="Four target mean absolute alpha changes: slight,moderate,large,extreme",
    )
    parser.add_argument(
        "--methods",
        default=",".join(TIME_VARYING_METHODS),
        help="Comma-separated methods. The four default ProbDPP variants are required for plotting.",
    )

    parser.add_argument("--gram-ridge", type=float, default=DEFAULT_GRAM_RIDGE)
    parser.add_argument("--kl-ucb-c", type=float, default=DEFAULT_KL_UCB_C)
    parser.add_argument("--online-index", choices=["kl_ucb", "mean"], default="kl_ucb")
    parser.add_argument("--mean-prior", type=float, default=0.7)
    parser.add_argument(
        "--adaptive-window",
        type=int,
        default=DEFAULT_ADAPTIVE_WINDOW,
        help="Per-source feedback window for Adaptive ProbDPP.",
    )

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
        help="Randomly assign the ten passages to simulated source indices using a fixed per-question permutation.",
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
    if args.change_point <= 0 or args.change_point >= args.num_questions:
        raise ValueError("change-point must satisfy 0 < change-point < num-questions")
    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if args.adaptive_window <= 0:
        raise ValueError("adaptive-window must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    alpha = parse_alpha_vector(args.alpha_vector, args.n, "--alpha-vector")
    deltas = parse_float_csv(args.severity_deltas, "--severity-deltas")
    build_severity_vectors(alpha, deltas)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
