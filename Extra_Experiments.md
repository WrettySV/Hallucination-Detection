# Extra Experiments and Work Log

This file summarizes extra work we tried during the project. It is not part of the main final notebook, but it shows what was explored and achieved.

## 1. Gemma Dataset Generation

We first generated a smaller hallucination dataset using Gemma. This was useful for testing the full pipeline before moving to the larger GPT-4o-mini/RAGTruth-style dataset.

Notebook:

- `extra_notebooks/toolace_hallucination_dataset_final.ipynb`

What we did:

- Extracted ToolACE tool-use examples.
- Generated hallucinations for contradiction, overgeneration, and missing-tool cases.
- Saved span labels in RAGTruth-style format.
- Validated spans with rule-based checks before saving.

Result:

- 200 rows per hallucination type were generated for the first stable dataset run.
- This helped confirm that the generation and validation pipeline worked before scaling.

## 2. Gemma Dataset Baseline Evaluation

Notebook:

- `extra_notebooks/toolace_detector_evaluation_final.ipynb`

We evaluated LettuceDetect and LookBackLens on the Gemma-generated dataset using token-level/span-level evaluation.

Gemma dataset results:

| Method | Dataset | F1 |
|---|---|---:|
| LettuceDetect | Overall | 0.4467 |
| LookBackLens | Overall | 0.4194 |

LettuceDetect was slightly better overall on this generated Gemma dataset.

## 3. GPT-4o-mini Dataset Baseline Evaluation

Notebook:

- `extra_notebooks/toolace_detector_evaluation_gpt4omini.ipynb`

We also evaluated the baselines on the larger GPT-4o-mini generated RAGTruth-style dataset.

GPT-4o-mini results:

| Method | Dataset | F1 |
|---|---|---:|
| LettuceDetect | Overall | 0.4534 |
| LookBackLens | Overall | 0.5565 |

On this dataset, LookBackLens-style attention features worked better than LettuceDetect.

## 4. NLI-Based Verifier Attempt

Notebook:

- `extra_notebooks/toolace_context_nli_verifier_gpt4omini.ipynb`

This was an attempt to improve the baseline by using an NLI model. We split each answer into sentences and checked whether each sentence was supported by the clean/reference answer using `microsoft/deberta-large-mnli`.

If the entailment probability was low, the sentence was marked as hallucinated and converted into character/token-level predictions.

NLI verifier results:

| Dataset | F1 |
|---|---:|
| Contradiction | 0.2458 |
| Overgeneration | 0.4551 |
| Missing tool | 0.7162 |
| Overall | 0.4693 |

The NLI verifier had high recall but low precision because it often marked whole sentences as hallucinated. It was useful as an experiment, but it was not better than the final tool-aware DeBERTa model.

