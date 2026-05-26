from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lettucedetect.models.inference import HallucinationDetector

MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
RAGTRUTH_DIR = Path("data/ragtruth_style")
CHECKPOINTS_DIR = Path("data/checkpoints")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_gold_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Gold spans for evaluator only — never passed to the detector."""
    labels = row.get("hallucination_labels", "[]")
    if isinstance(labels, str):
        labels = json.loads(labels)
    return [{"start": l["start"], "end": l["end"], "text": l["text"]} for l in labels]


def merge_char_positions_to_spans(output: str, positions: list[int]) -> list[dict[str, Any]]:
    if not positions:
        return []
    positions = sorted(set(positions))
    spans: list[dict[str, Any]] = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            spans.append({"start": start, "end": prev + 1, "text": output[start : prev + 1]})
            start = pos
            prev = pos
    spans.append({"start": start, "end": prev + 1, "text": output[start : prev + 1]})
    return spans


def tokens_to_char_positions(answer: str, tokens: list[dict[str, Any]], threshold: float) -> list[int]:
    """Align subword tokens to character indices (BPE-style, sequential search)."""
    positions: list[int] = []
    cursor = 0
    for tok in tokens:
        piece = tok["token"]
        if piece in ("[SEP]", "[CLS]", ""):
            break
        if tok.get("prob", 0.0) < threshold:
            idx = answer.find(piece, cursor)
            if idx == -1:
                idx = answer.find(piece.lstrip(), cursor)
            if idx != -1:
                cursor = idx + len(piece.lstrip() if piece.startswith(" ") else piece)
            continue
        idx = answer.find(piece, cursor)
        if idx == -1:
            idx = answer.find(piece.lstrip(), cursor)
        if idx == -1:
            continue
        length = len(piece) if piece in answer[idx : idx + len(piece)] else len(piece.lstrip())
        for i in range(idx, idx + length):
            positions.append(i)
        cursor = idx + length
    return positions


class LettuceDetectWrapper:
    """row -> predicted hallucination spans inside row['output']."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        threshold: float = 0.5,
        output_mode: str = "spans",
    ):
        self.threshold = threshold
        self.output_mode = output_mode
        self.detector = HallucinationDetector(method="transformer", model_path=model_path)

    def predict(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        question = row["query"]
        context = [row["context"]]
        answer = row["output"]

        if self.output_mode == "spans":
            raw = self.detector.predict(
                context=context,
                question=question,
                answer=answer,
                output_format="spans",
            )
            return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in raw]

        tokens = self.detector.predict(
            context=context,
            question=question,
            answer=answer,
            output_format="tokens",
        )
        positions = tokens_to_char_positions(answer, tokens, self.threshold)
        return merge_char_positions_to_spans(answer, positions)

def trim_span_whitespace(output: str, span: dict[str, Any]) -> dict[str, Any]:
    """Remove leading/trailing whitespace from one span while keeping offsets valid."""
    start = span["start"]
    end = span["end"]

    while start < end and output[start].isspace():
        start += 1

    while end > start and output[end - 1].isspace():
        end -= 1

    return {
        **span,
        "start": start,
        "end": end,
        "text": output[start:end],
    }

def trim_spans_whitespace(output: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim whitespace around all spans."""
    return [trim_span_whitespace(output, span) for span in spans]

def char_level_prf(
    pred_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    text_len: int,
) -> dict[str, float]:
    def mask(spans: list[dict[str, Any]]) -> list[int]:
        m = [0] * text_len
        for s in spans:
            for i in range(max(0, s["start"]), min(text_len, s["end"])):
                m[i] = 1
        return m

    pred_m = mask(pred_spans)
    gold_m = mask(gold_spans)
    tp = sum(p & g for p, g in zip(pred_m, gold_m))
    fp = sum(p & (1 - g) for p, g in zip(pred_m, gold_m))
    fn = sum((1 - p) & g for p, g in zip(pred_m, gold_m))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def predict_all(
    rows: list[dict[str, Any]],
    detector: LettuceDetectWrapper,
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

            iterator = tqdm(subset, desc="LettuceDetect", unit="row")
        except ImportError:
            pass

    for row in iterator:
        rid = str(row["id"])
        output = row["output"]
        predictions[rid] = trim_spans_whitespace(output, detector.predict(row))
    return predictions


def save_predictions(
    predictions: dict[str, list[dict[str, Any]]],
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write {row_id: pred_spans} to JSONL (one record per line: id, pred_spans)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rid in sorted(predictions):
            f.write(
                json.dumps({"id": rid, "pred_spans": predictions[rid]}, ensure_ascii=False)
                + "\n"
            )
    if meta is not None:
        meta_path = path.with_name(path.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_predictions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load predictions written by save_predictions."""
    path = Path(path)
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["id"])] = rec.get("pred_spans", [])
    return out


def load_predictions_if_exists(path: str | Path) -> dict[str, list[dict[str, Any]]] | None:
    path = Path(path)
    if not path.exists():
        return None
    return load_predictions(path)


def evaluate_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Micro character-level P/R/F1 over a fixed prediction dict (no re-inference)."""
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        output = row["output"]
        rid = str(row["id"])
        gold_spans = trim_spans_whitespace(output, parse_gold_spans(row))
        pred_spans = trim_spans_whitespace(output, predictions.get(rid, []))
        m = char_level_prf(pred_spans, gold_spans, len(output))
        totals["tp"] += m["tp"]
        totals["fp"] += m["fp"]
        totals["fn"] += m["fn"]

    precision = (
        totals["tp"] / (totals["tp"] + totals["fp"])
        if (totals["tp"] + totals["fp"])
        else 0.0
    )
    recall = (
        totals["tp"] / (totals["tp"] + totals["fn"])
        if (totals["tp"] + totals["fn"])
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(rows)}


def summarize_mixed_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count clean vs corrupted rows and corrupted rows per hallucination type."""
    clean_rows = [r for r in rows if str(r.get("hallucination_type")) == "clean"]
    by_type: dict[str, int] = {}
    for row in rows:
        htype = str(row.get("hallucination_type", "unknown"))
        if htype == "clean":
            continue
        by_type[htype] = by_type.get(htype, 0) + 1
    n_corrupted = len(rows) - len(clean_rows)
    return {
        "n_total": len(rows),
        "n_clean": len(clean_rows),
        "n_corrupted": n_corrupted,
        "by_type": dict(sorted(by_type.items())),
    }


DEFAULT_CORRUPTION_TYPES = (
    "contradiction",
    "missing_tool",
    "overgeneration",
)


def evaluate_corruption_type_mix(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    corruption_types: tuple[str, ...] = DEFAULT_CORRUPTION_TYPES,
) -> list[dict[str, Any]]:
    """
    Per corruption type: micro P/R/F1 on (hallucinated rows of that type) + (all test clean).

    Matches the Pipeline type-eval idea: each slice includes clean negatives, without a
    separate standalone 'clean' metrics row.
    """
    clean_rows = [r for r in rows if str(r.get("hallucination_type")) == "clean"]
    n_clean = len(clean_rows)

    out: list[dict[str, Any]] = []
    for htype in corruption_types:
        hall_rows = [r for r in rows if str(r.get("hallucination_type")) == htype]
        subset = hall_rows + clean_rows
        m = evaluate_predictions(subset, predictions)
        m["hallucination_type"] = htype
        m["n_corrupted"] = len(hall_rows)
        m["n_clean"] = n_clean
        out.append(m)
    return out


def evaluate_by_hallucination_type(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    type_field: str = "hallucination_type",
) -> list[dict[str, Any]]:
    """Per-type metrics from the same predictions (one group per label, incl. clean)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(type_field) or "unknown")
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for htype in sorted(groups):
        m = evaluate_predictions(groups[htype], predictions)
        m["hallucination_type"] = htype
        out.append(m)
    return out


def evaluate_lettucedetect(
    rows: list[dict[str, Any]],
    detector: LettuceDetectWrapper,
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
