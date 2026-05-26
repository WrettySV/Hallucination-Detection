# Hallucination Detection in Tool Calling

This repository contains the final project files for the assignment **Hallucination Detection in Tool Calling**.

## What This Project Does

The task is to detect hallucinated spans in tool-calling dialogue answers. The work follows three main steps:

1. Generate a hallucination dataset from ToolACE.
2. Evaluate required baselines: LettuceDetect and LookBackLens.
3. Improve the baselines with a tool-aware DeBERTa token classifier.

The labels are span-based and follow a RAGTruth-style format:

- `query` = user question
- `context` = tool output
- `output` = assistant final answer
- `hallucination_labels` = hallucinated character spans

## Files

| File | Purpose |
|---|---|
| `final_notebook_hallucination_detection.ipynb` | Main project notebook with dataset generation, baseline evaluation, and improved model experiment. |
| `final_notebook_hallucination_detection_sumbission.ipynb` | Submission-style notebook. The filename has a typo in `sumbission`, but the file is kept unchanged. |
| `lettuce_baseline.py` | Helper code for LettuceDetect baseline evaluation. |
| `lookback_baseline.py` | Helper code for the LookBackLens-style attention baseline. |
| `toolaware_deberta_baseline.py` | Helper code for the improved tool-aware DeBERTa model. |
| `Extra_Experiments.md` | Notes about extra experiments, including Gemma dataset generation/evaluation and the NLI verifier attempt. |

## Main Result

The improved tool-aware DeBERTa model gives the best reported result.

| Method | Character F1 |
|---|---:|
| LettuceDetect | 0.291 |
| LookBackLens-style | 0.443 |
| Tool-aware DeBERTa | 0.945 |

Per hallucination type:

| Type | Tool-aware DeBERTa F1 |
|---|---:|
| Contradiction | 0.651 |
| Missing tool | 0.978 |
| Overgeneration | 0.956 |

## Notes

- The final notebooks already contain executed outputs.
- The helper scripts are included for reproducibility.
- `Extra_Experiments.md` mentions extra notebooks, but those notebook files are not present in this cloned repository.
- Existing files were not modified when this README was added.

