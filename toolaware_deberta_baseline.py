"""Tool-aware DeBERTa-v3-small span detector for mixed ToolACE splits."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lettuce_baseline import CHECKPOINTS_DIR, parse_gold_spans, trim_spans_whitespace

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base" #"microsoft/deberta-v3-small"
DEFAULT_RUN_NAME = "toolaware_deberta_mixed"


def patch_transformers_for_text_only() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers.utils.import_utils as import_utils

        if hasattr(import_utils, "_torchvision_available"):
            import_utils._torchvision_available = False
        if hasattr(import_utils, "_torchvision_version"):
            import_utils._torchvision_version = "N/A"
    except Exception:
        pass


def compact_json(x: Any, max_chars: int = 800) -> str:
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s[:max_chars] + " ...[truncated]" if len(s) > max_chars else s


def build_context_prompt(row: dict[str, Any]) -> str:
    tools = row.get("available_tools_json") or row.get("available_tools") or []
    tool_names = row.get("tool_names") or tools
    return (
        "[QUERY]\n"
        f"{row.get('query', '')}\n\n"
        "[AVAILABLE_TOOLS]\n"
        f"{compact_json(tools, 800)}\n\n"
        "[TOOL_NAMES]\n"
        f"{compact_json(tool_names, 400)}\n\n"
        "[TOOL_CALL]\n"
        f"{compact_json(row.get('tool_call', ''), 600)}\n\n"
        "[TOOL_OUTPUT]\n"
        f"{row.get('context', '')}"
    )


def make_char_mask(answer: str, labels: list[dict[str, Any]]) -> list[int]:
    mask = [0] * len(answer)
    for lab in labels or []:
        s, e = int(lab["start"]), int(lab["end"])
        s = max(0, min(s, len(answer)))
        e = max(0, min(e, len(answer)))
        for i in range(s, e):
            mask[i] = 1
    return mask


def merge_positions_to_spans(answer: str, positions: list[int]) -> list[dict[str, Any]]:
    positions = sorted(set(int(p) for p in positions if 0 <= int(p) < len(answer)))
    if not positions:
        return []
    spans: list[dict[str, Any]] = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            spans.append(
                {"start": start, "end": prev + 1, "text": answer[start : prev + 1]}
            )
            start = pos
            prev = pos
    spans.append({"start": start, "end": prev + 1, "text": answer[start : prev + 1]})
    return spans


def is_contradiction_row(row: dict[str, Any]) -> bool:
    label = str(
        row.get("hallucination_type")
        or row.get("variant")
        or row.get("corruption_type")
        or ""
    ).lower()
    return "contradiction" in label


def oversample_contradiction_rows(
    rows: list[dict[str, Any]],
    factor: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Repeat contradiction training rows so the hardest type contributes more updates."""
    if factor <= 1:
        return list(rows)

    contradictions = [row for row in rows if is_contradiction_row(row)]
    out = list(rows) + contradictions * (factor - 1)
    random.Random(seed).shuffle(out)
    return out


@dataclass
class ToolAwareDeBERTaConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    num_epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    max_pos_class_weight: float = 20.0
    contradiction_oversample: int = 1
    seed: int = 42
    thresholds: list[float] = field(
        default_factory=lambda: [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    )


class ToolAwareDeBERTaDetector:
    """Fine-tuned DeBERTa token classifier with tool-aware context prompt."""

    def __init__(
        self,
        config: ToolAwareDeBERTaConfig | None = None,
        *,
        model_dir: str | Path | None = None,
        threshold: float = 0.5,
    ):
        self.config = config or ToolAwareDeBERTaConfig()
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if self.model_dir is None or not self.model_dir.exists():
            raise ValueError("Model not trained. Call train() or set model_dir to saved checkpoint.")

        patch_transformers_for_text_only()
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        self._lazy_load()
        scores = self._score_row_chars(row)
        answer = row.get("output", "")
        positions = [i for i, p in enumerate(scores) if p >= self.threshold]
        spans = merge_positions_to_spans(answer, positions)
        return trim_spans_whitespace(answer, spans)

    def _score_row_chars(self, row: dict[str, Any]) -> np.ndarray:
        import torch
        from transformers import AutoTokenizer

        self._lazy_load()
        assert self._tokenizer is not None and self._model is not None

        context_prompt = build_context_prompt(row)
        answer = row.get("output", "")
        cfg = self.config

        try:
            enc = self._tokenizer(
                context_prompt,
                answer,
                truncation="only_first",
                max_length=cfg.max_length,
                return_offsets_mapping=True,
            )
        except Exception:
            enc = self._tokenizer(
                context_prompt,
                answer,
                truncation=True,
                max_length=cfg.max_length,
                return_offsets_mapping=True,
            )

        seq_ids = enc.sequence_ids()
        offsets = enc["offset_mapping"]
        inputs = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor(
                [enc["attention_mask"]], dtype=torch.long, device=self._device
            ),
        }
        if "token_type_ids" in enc:
            inputs["token_type_ids"] = torch.tensor(
                [enc["token_type_ids"]], dtype=torch.long, device=self._device
            )

        use_amp = self._device.type == "cuda"
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0, :, 1].detach().cpu().numpy()

        char_scores = np.zeros(len(answer), dtype=np.float32)
        for i, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
            if seq_id != 1:
                continue
            start, end = int(offset[0]), int(offset[1])
            if end <= start:
                continue
            start = max(0, min(start, len(answer)))
            end = max(0, min(end, len(answer)))
            if end > start:
                char_scores[start:end] = np.maximum(char_scores[start:end], probs[i])
        return char_scores

    def train(
        self,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        *,
        model_dir: str | Path,
        max_train_rows: int | None = None,
        max_val_rows: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Train on mixed_train, pick threshold on mixed_val, save model + meta."""
        patch_transformers_for_text_only()
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            get_linear_schedule_with_warmup,
        )

        from lettuce_baseline import evaluate_corruption_type_mix, evaluate_predictions

        cfg = self.config
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_subset_base = train_rows if max_train_rows is None else train_rows[:max_train_rows]
        train_subset = oversample_contradiction_rows(
            train_subset_base,
            cfg.contradiction_oversample,
            seed=cfg.seed,
        )
        val_subset = val_rows if max_val_rows is None else val_rows[:max_val_rows]

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer required for offset mappings.")

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_name,
            num_labels=2,
            id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
            label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
        )
        model.to(device)

        def encode_record(row: dict[str, Any]) -> dict[str, Any]:
            context_prompt = build_context_prompt(row)
            answer = row.get("output", "")
            char_mask = make_char_mask(answer, parse_gold_spans(row))
            try:
                enc = tokenizer(
                    context_prompt,
                    answer,
                    truncation="only_first",
                    max_length=cfg.max_length,
                    return_offsets_mapping=True,
                )
            except Exception:
                enc = tokenizer(
                    context_prompt,
                    answer,
                    truncation=True,
                    max_length=cfg.max_length,
                    return_offsets_mapping=True,
                )
            seq_ids = enc.sequence_ids()
            offsets = enc["offset_mapping"]
            token_labels: list[int] = []
            for seq_id, (start, end) in zip(seq_ids, offsets):
                if seq_id != 1 or end <= start:
                    token_labels.append(-100)
                    continue
                start = max(0, min(int(start), len(answer)))
                end = max(0, min(int(end), len(answer)))
                token_labels.append(1 if any(char_mask[start:end]) else 0)
            enc.pop("offset_mapping")
            enc["labels"] = token_labels
            return enc

        def encode_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
            iterator: Any = rows
            if show_progress:
                iterator = tqdm(rows, desc=f"Encoding {name}")
            return [encode_record(r) for r in iterator]

        train_encoded = encode_rows(train_subset, "train")
        val_encoded = encode_rows(val_subset, "validation")

        class TokenDataset(Dataset):
            def __init__(self, encoded_rows: list[dict[str, Any]]):
                self.items = []
                for enc in encoded_rows:
                    item = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": enc["labels"],
                    }
                    if "token_type_ids" in enc:
                        item["token_type_ids"] = enc["token_type_ids"]
                    self.items.append(item)

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return self.items[idx]

        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding=True, return_tensors="pt"
        )
        train_loader = DataLoader(
            TokenDataset(train_encoded),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            TokenDataset(val_encoded),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        label_counts: Counter[int] = Counter()
        for enc in train_encoded:
            for y in enc["labels"]:
                if y in (0, 1):
                    label_counts[int(y)] += 1
        neg_count = label_counts[0]
        pos_count = label_counts[1]
        pos_weight = min(float(neg_count / max(1, pos_count)), cfg.max_pos_class_weight)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
        total_training_steps = cfg.num_epochs * num_update_steps_per_epoch
        warmup_steps = int(cfg.warmup_ratio * total_training_steps)

        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_training_steps
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def score_row_chars_current(row: dict[str, Any]) -> np.ndarray:
            context_prompt = build_context_prompt(row)
            answer = row.get("output", "")
            try:
                enc = tokenizer(
                    context_prompt,
                    answer,
                    truncation="only_first",
                    max_length=cfg.max_length,
                    return_offsets_mapping=True,
                )
            except Exception:
                enc = tokenizer(
                    context_prompt,
                    answer,
                    truncation=True,
                    max_length=cfg.max_length,
                    return_offsets_mapping=True,
                )

            seq_ids = enc.sequence_ids()
            offsets = enc["offset_mapping"]
            inputs = {
                "input_ids": torch.tensor(
                    [enc["input_ids"]], dtype=torch.long, device=device
                ),
                "attention_mask": torch.tensor(
                    [enc["attention_mask"]], dtype=torch.long, device=device
                ),
            }
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = torch.tensor(
                    [enc["token_type_ids"]], dtype=torch.long, device=device
                )

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(**inputs)
                    probs = (
                        torch.softmax(outputs.logits, dim=-1)[0, :, 1]
                        .detach()
                        .cpu()
                        .numpy()
                    )

            char_scores = np.zeros(len(answer), dtype=np.float32)
            for i, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
                if seq_id != 1:
                    continue
                start, end = int(offset[0]), int(offset[1])
                if end <= start:
                    continue
                start = max(0, min(start, len(answer)))
                end = max(0, min(end, len(answer)))
                if end > start:
                    char_scores[start:end] = np.maximum(char_scores[start:end], probs[i])
            return char_scores

        def predictions_from_scores(
            rows: list[dict[str, Any]],
            scores: dict[str, np.ndarray],
            threshold: float,
        ) -> dict[str, list[dict[str, Any]]]:
            return {
                str(row["id"]): trim_spans_whitespace(
                    row["output"],
                    merge_positions_to_spans(
                        row["output"],
                        [
                            i
                            for i, p in enumerate(scores[str(row["id"])])
                            if p >= threshold
                        ],
                    ),
                )
                for row in rows
            }

        def search_threshold(
            rows: list[dict[str, Any]],
            scores: dict[str, np.ndarray],
        ) -> tuple[
            list[dict[str, Any]],
            float,
            dict[str, float],
            dict[str, list[dict[str, Any]]],
        ]:
            rows_by_threshold = []
            preds_by_threshold = {}
            for thr in cfg.thresholds:
                preds = predictions_from_scores(rows, scores, thr)
                metrics = evaluate_predictions(rows, preds)
                rows_by_threshold.append({"threshold": thr, **metrics})
                preds_by_threshold[thr] = preds

            best_row = sorted(
                rows_by_threshold,
                key=lambda x: (x["f1"], x["precision"], x["recall"]),
                reverse=True,
            )[0]
            best_thr = float(best_row["threshold"])
            return rows_by_threshold, best_thr, best_row, preds_by_threshold[best_thr]

        best_val_loss = float("inf")
        best_val_f1 = -1.0
        best_threshold = 0.5
        best_epoch: int | None = None
        train_log: list[dict[str, Any]] = []
        t0 = time.time()

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_batches = 0

            train_iter: Any = train_loader
            if show_progress:
                train_iter = tqdm(train_loader, desc=f"DeBERTa epoch {epoch}/{cfg.num_epochs}")

            for step, batch in enumerate(train_iter, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"),
                    )
                    loss = loss_fct(
                        outputs.logits.view(-1, 2), batch["labels"].view(-1)
                    )
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * cfg.grad_accum_steps
                running_batches += 1
                if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()
            val_losses: list[float] = []
            val_iter: Any = val_loader
            if show_progress:
                val_iter = tqdm(val_loader, desc="Validation loss", leave=False)
            with torch.no_grad():
                for batch in val_iter:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch.get("token_type_ids"),
                        )
                        loss = loss_fct(
                            outputs.logits.view(-1, 2), batch["labels"].view(-1)
                        )
                    val_losses.append(float(loss.detach().cpu()))
            train_loss = running_loss / max(1, running_batches)
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0

            model.eval()
            val_scores_epoch = {
                str(row["id"]): score_row_chars_current(row) for row in val_subset
            }
            _, epoch_threshold, epoch_metrics, epoch_preds = search_threshold(
                val_subset, val_scores_epoch
            )
            per_type_rows = evaluate_corruption_type_mix(val_subset, epoch_preds)
            contradiction_f1 = next(
                (
                    row["f1"]
                    for row in per_type_rows
                    if row["hallucination_type"] == "contradiction"
                ),
                0.0,
            )

            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, val_f1={epoch_metrics['f1']:.4f}, "
                f"threshold={epoch_threshold:.2f}, contradiction_f1={contradiction_f1:.4f}"
            )

            train_log.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_precision": epoch_metrics["precision"],
                    "val_recall": epoch_metrics["recall"],
                    "val_f1": epoch_metrics["f1"],
                    "val_threshold": epoch_threshold,
                    "val_contradiction_f1": contradiction_f1,
                }
            )
            is_better = (
                epoch_metrics["f1"] > best_val_f1
                or (
                    epoch_metrics["f1"] == best_val_f1
                    and val_loss < best_val_loss
                )
            )
            if is_better:
                best_val_loss = val_loss
                best_val_f1 = float(epoch_metrics["f1"])
                best_threshold = epoch_threshold
                best_epoch = epoch
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)

        training_seconds = time.time() - t0
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._lazy_load()

        val_scores = {str(r["id"]): self._score_row_chars(r) for r in val_subset}
        threshold_rows, best_thr, final_val_metrics, final_val_preds = search_threshold(
            val_subset, val_scores
        )
        self.threshold = best_thr
        final_val_per_type = evaluate_corruption_type_mix(val_subset, final_val_preds)

        meta = {
            "model_name": cfg.model_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_f1": final_val_metrics["f1"],
            "best_threshold": best_thr,
            "pos_weight": float(pos_weight),
            "training_seconds": training_seconds,
            "n_train": len(train_subset_base),
            "n_train_after_oversample": len(train_subset),
            "n_validation": len(val_subset),
            "contradiction_oversample": cfg.contradiction_oversample,
            "train_log": train_log,
            "threshold_search": threshold_rows,
            "validation_per_type": final_val_per_type,
        }
        meta_path = model_dir / "run_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, model_dir: str | Path) -> ToolAwareDeBERTaDetector:
        model_dir = Path(model_dir)
        meta_path = model_dir / "run_meta.json"
        threshold = 0.5
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            threshold = float(meta.get("best_threshold", threshold))
        det = cls(model_dir=model_dir, threshold=threshold)
        return det


def predict_all(
    rows: list[dict[str, Any]],
    detector: ToolAwareDeBERTaDetector,
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

            iterator = tqdm(subset, desc="Tool-aware DeBERTa", unit="row")
        except ImportError:
            pass

    for row in iterator:
        rid = str(row["id"])
        output = row["output"]
        predictions[rid] = trim_spans_whitespace(output, detector.predict(row))
    return predictions


def default_model_dir(run_name: str = DEFAULT_RUN_NAME) -> Path:
    return CHECKPOINTS_DIR / f"{run_name}_model"


def default_preds_path(run_name: str = DEFAULT_RUN_NAME) -> Path:
    return CHECKPOINTS_DIR / f"{run_name}_test_predictions.jsonl"
