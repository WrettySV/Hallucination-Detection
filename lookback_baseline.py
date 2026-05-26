from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForCausalLM, AutoTokenizer

from lettuce_baseline import (
    CHECKPOINTS_DIR,
    char_level_prf,
    evaluate_corruption_type_mix,
    evaluate_predictions,
    load_predictions,
    load_predictions_if_exists,
    merge_char_positions_to_spans,
    parse_gold_spans,
    save_predictions,
    summarize_mixed_test,
    trim_spans_whitespace,
)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PAPER_MODEL = "meta-llama/Llama-2-7b-chat-hf"
PromptStyle = Literal["toolace", "paper_nq"]


def train_test_split_rows(
    rows: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split rows (same pattern as official step03 train_test_split)."""
    rows = rows.copy()
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_ratio))
    return rows[n_test:], rows[:n_test]


def build_prompt(
    query: str,
    context: str,
    answer: str,
    style: PromptStyle = "toolace",
) -> tuple[str, str]:
    """Return (prefix_before_answer, full_text)."""
    if style == "paper_nq":
        # Roughly aligned with Lookback-Lens NQ prompt (#Document# / #Question# / #Answer#).
        prefix = f"#Document#: {context}\n#Question#: {query}\n#Answer#:"
    else:
        prefix = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:\n"
        )
    return prefix, prefix + answer


def char_hallucination_mask(output: str, gold_spans: list[dict[str, Any]]) -> list[int]:
    """1 = factual, 0 = hallucinated (character level)."""
    mask = [1] * len(output)
    for span in gold_spans:
        for i in range(max(0, span["start"]), min(len(output), span["end"])):
            mask[i] = 0
    return mask


def token_labels_from_chars(
    output: str,
    offset_mapping: list[tuple[int, int]],
    answer_char_start: int,
    char_mask: list[int],
) -> tuple[list[int], list[int]]:
    """Map char labels to answer token indices. Returns (token_indices, labels)."""
    token_indices: list[int] = []
    labels: list[int] = []
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == end == 0:
            continue
        if end <= answer_char_start:
            continue
        a_start = max(0, start - answer_char_start)
        a_end = min(len(output), end - answer_char_start)
        if a_start >= a_end:
            continue
        chars = char_mask[a_start:a_end]
        label = 1 if sum(chars) > len(chars) / 2 else 0
        token_indices.append(tok_idx)
        labels.append(label)
    return token_indices, labels


def compute_lookback_ratios(
    attentions: tuple[torch.Tensor, ...],
    answer_start: int,
    token_indices: list[int],
) -> np.ndarray:
    """
    Lookback ratio per (layer, head), concatenated — as in voidism/Lookback-Lens step01:

        ratio = attn(context) / (attn(context) + attn(generated_answer_so_far))
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    n_features = n_layers * n_heads
    features = np.zeros((len(token_indices), n_features), dtype=np.float32)
    eps = 1e-8

    for row_i, seq_i in enumerate(token_indices):
        feat_col = 0
        for layer_attn in attentions:
            attn = layer_attn[0]
            for head in range(n_heads):
                weights = attn[head, seq_i, : seq_i + 1]
                ctx = weights[:answer_start].sum().item()
                gen = weights[answer_start : seq_i + 1].sum().item()
                features[row_i, feat_col] = ctx / (ctx + gen + eps)
                feat_col += 1
    return features


class LookBackLensWrapper:
    """row -> predicted hallucination spans inside row['output']."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.5,
        device: str | None = None,
        max_length: int = 2048,
        sliding_window: int = 8,
        classifier_path: str | Path | None = None,
        prompt_style: PromptStyle = "toolace",
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.max_length = max_length
        self.sliding_window = max(1, sliding_window)
        self.prompt_style = prompt_style
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        self.classifier: LogisticRegression | SGDClassifier | None = None
        if classifier_path is not None:
            self.classifier = joblib.load(classifier_path)

    def _encode(self, query: str, context: str, answer: str) -> tuple[dict[str, Any], int, list[tuple[int, int]]]:
        prefix, full = build_prompt(query, context, answer, style=self.prompt_style)
        encoded = self.tokenizer(
            full,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        offset_mapping = [tuple(m) for m in encoded.pop("offset_mapping")[0].tolist()]
        answer_char_start = len(prefix)

        # First token whose span starts at/after answer text.
        answer_start = len(offset_mapping)
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0:
                continue
            if start >= answer_char_start:
                answer_start = i
                break

        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded, answer_start, offset_mapping

    def _lookback_features(
        self,
        row: dict[str, Any],
        gold_spans: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, list[int], list[int], list[tuple[int, int]], str]:
        query, context, answer = row["query"], row["context"], row["output"]
        encoded, answer_start, offset_mapping = self._encode(query, context, answer)

        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True)

        char_mask = char_hallucination_mask(answer, gold_spans or [])
        prefix = build_prompt(query, context, answer, style=self.prompt_style)[0]
        token_indices, labels = token_labels_from_chars(
            answer, offset_mapping, len(prefix), char_mask
        )
        if not token_indices:
            return np.zeros((0, 0), dtype=np.float32), [], [], offset_mapping, answer

        features = compute_lookback_ratios(outputs.attentions, answer_start, token_indices)
        if self.sliding_window > 1 and len(features):
            features = _sliding_window_average(features, self.sliding_window)
        return features, labels, token_indices, offset_mapping, answer

    def _collect_token_features(
        self,
        rows: list[dict[str, Any]],
        *,
        max_rows: int | None = None,
        desc: str = "Extract features",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack token-level lookback features and labels (1=factual, 0=hallucinated)."""
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x, **_: x  # type: ignore

        subset = rows if max_rows is None else rows[:max_rows]
        x_parts: list[np.ndarray] = []
        y_parts: list[int] = []

        for row in tqdm(subset, desc=desc, unit="row"):
            gold = parse_gold_spans(row)
            features, labels, _, _, _ = self._lookback_features(row, gold_spans=gold)
            if len(labels) == 0:
                continue
            x_parts.append(features)
            y_parts.extend(labels)

        if not x_parts:
            raise ValueError("No tokens collected; check rows / max_rows.")

        x = np.vstack(x_parts)
        y = np.array(y_parts, dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError("Need both factual and hallucinated tokens.")
        return x, y

    @staticmethod
    def _save_feature_cache(path: str | Path, x: np.ndarray, y: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=x, y=y)

    @staticmethod
    def _load_feature_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        bundle = np.load(Path(path))
        return bundle["x"], bundle["y"]

    @staticmethod
    def _token_split_metrics(
        classifier: SGDClassifier | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        pred = classifier.predict(x)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        }

    def fit(
        self,
        rows: list[dict[str, Any]],
        max_rows: int | None = 200,
    ) -> None:
        """Train logistic regression on token-level lookback-ratio features (single pass)."""
        x, y = self._collect_token_features(rows, max_rows=max_rows, desc="LookBackLens fit")
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.classifier.fit(x, y)

    def fit_with_validation(
        self,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        *,
        epochs: int = 20,
        max_train_rows: int | None = None,
        max_val_rows: int | None = None,
        train_cache_path: str | Path | None = None,
        val_cache_path: str | Path | None = None,
    ) -> None:
        """
        Train SGD logistic regression with epoch-wise token metrics on train and validation.

        Attention features are extracted once per split, then the linear head is updated
        for ``epochs`` passes (similar in spirit to monitoring train/val during training).
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        if train_cache_path is not None and Path(train_cache_path).exists():
            x_train, y_train = self._load_feature_cache(train_cache_path)
            print(f"Loaded LookBackLens train feature cache: {Path(train_cache_path).resolve()}")
        else:
            x_train, y_train = self._collect_token_features(
                train_rows,
                max_rows=max_train_rows,
                desc="LookBackLens train features",
            )
            if train_cache_path is not None:
                self._save_feature_cache(train_cache_path, x_train, y_train)
                print(f"Saved LookBackLens train feature cache: {Path(train_cache_path).resolve()}")

        if val_cache_path is not None and Path(val_cache_path).exists():
            x_val, y_val = self._load_feature_cache(val_cache_path)
            print(f"Loaded LookBackLens val feature cache: {Path(val_cache_path).resolve()}")
        else:
            x_val, y_val = self._collect_token_features(
                val_rows,
                max_rows=max_val_rows,
                desc="LookBackLens val features",
            )
            if val_cache_path is not None:
                self._save_feature_cache(val_cache_path, x_val, y_val)
                print(f"Saved LookBackLens val feature cache: {Path(val_cache_path).resolve()}")

        classes = np.array([0, 1], dtype=np.int64)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        clf = SGDClassifier(
            loss="log_loss",
            class_weight=class_weight,
            random_state=42,
            warm_start=True,
            max_iter=1,
            learning_rate="optimal",
        )

        print(
            f"Token samples: train={len(y_train)} (factual={int((y_train == 1).sum())}, "
            f"hallucinated={int((y_train == 0).sum())})  "
            f"val={len(y_val)} (factual={int((y_val == 1).sum())}, "
            f"hallucinated={int((y_val == 0).sum())})"
        )
        print(f"{'epoch':>8}  {'train_acc':>10}  {'train_bal':>10}  {'val_acc':>10}  {'val_bal':>10}")

        for epoch in range(1, epochs + 1):
            clf.partial_fit(x_train, y_train, classes=classes)
            tr = self._token_split_metrics(clf, x_train, y_train)
            va = self._token_split_metrics(clf, x_val, y_val)
            print(
                f"{epoch:8d}  {tr['accuracy']:10.4f}  {tr['balanced_accuracy']:10.4f}  "
                f"{va['accuracy']:10.4f}  {va['balanced_accuracy']:10.4f}"
            )

        self.classifier = clf

    def save_classifier(self, path: str | Path) -> None:
        """Save only the sklearn classifier (small file). LLM weights stay on HuggingFace."""
        self.save_artifacts(path)

    def save_artifacts(self, path: str | Path) -> None:
        """Save classifier + settings needed to reload without refitting."""
        if self.classifier is None:
            raise ValueError("Classifier not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "model_name": self.model_name,
                "threshold": self.threshold,
                "max_length": self.max_length,
                "sliding_window": self.sliding_window,
                "prompt_style": self.prompt_style,
            },
            path,
        )

    @classmethod
    def load_artifacts(
        cls,
        path: str | Path,
        device: str | None = None,
    ) -> "LookBackLensWrapper":
        """Load saved classifier; downloads LLM from HuggingFace once."""
        bundle = joblib.load(path)
        detector = cls(
            model_name=bundle["model_name"],
            threshold=bundle["threshold"],
            max_length=bundle["max_length"],
            sliding_window=bundle["sliding_window"],
            prompt_style=bundle["prompt_style"],
            device=device,
        )
        detector.classifier = bundle["classifier"]
        return detector

    def predict(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        if self.classifier is None:
            raise ValueError("Call fit() first or pass classifier_path=...")

        features, _, token_indices, offset_mapping, answer = self._lookback_features(
            row, gold_spans=None
        )
        if len(features) == 0:
            return []

        # Class 1 = factual; mark hallucination when P(factual) < threshold.
        proba_factual = self.classifier.predict_proba(features)[:, 1]
        hallu_char_positions: list[int] = []

        answer_char_start = len(
            build_prompt(row["query"], row["context"], answer, style=self.prompt_style)[0]
        )
        for tok_i, seq_i in zip(range(len(token_indices)), token_indices):
            if proba_factual[tok_i] >= self.threshold:
                continue
            start, end = offset_mapping[seq_i]
            a_start = max(0, start - answer_char_start)
            a_end = min(len(answer), end - answer_char_start)
            for c in range(a_start, a_end):
                hallu_char_positions.append(c)

        return merge_char_positions_to_spans(answer, sorted(set(hallu_char_positions)))


def predict_all(
    rows: list[dict[str, Any]],
    detector: LookBackLensWrapper,
    *,
    max_rows: int | None = None,
    show_progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Run detector once per row; return {row_id: pred_spans}."""
    subset = rows if max_rows is None else rows[:max_rows]
    predictions: dict[str, list[dict[str, Any]]] = {}

    iterator: Any = subset
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(subset, desc="LookBackLens predict", unit="row")
        except ImportError:
            pass

    for row in iterator:
        rid = str(row["id"])
        output = row["output"]
        predictions[rid] = trim_spans_whitespace(output, detector.predict(row))
    return predictions


def _sliding_window_average(features: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(features)
    for i in range(len(features)):
        lo = max(0, i - window + 1)
        out[i] = features[lo : i + 1].mean(axis=0)
    return out


def evaluate_lookbacklens(
    rows: list[dict[str, Any]],
    detector: LookBackLensWrapper,
    max_rows: int | None = None,
) -> dict[str, float]:
    subset = rows if max_rows is None else rows[:max_rows]
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for row in subset:
        pred_spans = detector.predict(row)
        gold_spans = parse_gold_spans(row)
        pred_spans = trim_spans_whitespace(row["output"], pred_spans)
        gold_spans = trim_spans_whitespace(row["output"], gold_spans)
        m = char_level_prf(pred_spans, gold_spans, len(row["output"]))
        totals["tp"] += m["tp"]
        totals["fp"] += m["fp"]
        totals["fn"] += m["fn"]
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    r = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "n": len(subset)}


def fit_and_evaluate_lookbacklens(
    rows: list[dict[str, Any]],
    detector: LookBackLensWrapper,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    save_artifacts_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train on train split, evaluate on disjoint test split (recommended for reporting)."""
    train_rows, test_rows = train_test_split_rows(rows, test_ratio=test_ratio, seed=seed)
    if max_train_rows is not None:
        train_rows = train_rows[:max_train_rows]
    detector.fit(train_rows, max_rows=None)
    if save_artifacts_path is not None:
        detector.save_artifacts(save_artifacts_path)
    metrics = evaluate_lookbacklens(test_rows, detector, max_rows=max_test_rows)
    return {
        **metrics,
        "n_train": len(train_rows),
        "n_test": metrics["n"],
        "test_ratio": test_ratio,
        "seed": seed,
        "artifacts_path": str(save_artifacts_path) if save_artifacts_path else None,
    }
