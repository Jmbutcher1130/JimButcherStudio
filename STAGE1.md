# Calibrating Tiny Aya to distinguish safe and unsafe requests

Can a small calibration set help an open language model recognize unsafe requests without flagging more harmless ones? We ran Cohere Labs Tiny Aya Global locally and fitted a decision threshold using 24 human-reviewed requests. We evaluated the original and fitted thresholds on the same 36 XSTest requests using Lexsi AuditKIT 1.1.9. Accuracy changed from 33/36 to 23/36. The calibrated classifier identified 16/16 unsafe requests and correctly cleared 7/20 benign requests. We release the fitted classifier head, per-case scores and reproduction code. The evaluation measures request classification. The effect of steering on open-ended answers remains unmeasured.

## Auditing the decision boundary

With the base model held fixed, we asked whether changing the cutoff for an unsafe label would catch more unsafe requests and how many benign requests it would block. Every decision comes from a forward pass through the pinned CohereLabs/tiny-aya-global model. The English classification prompt ends with a choice between A (safe) and B (unsafe). The score is the next-token logit for “ B” minus the logit for “ A”. No free-text response is generated.

We used 12 safe and 12 unsafe calibration requests. The reviewer saw the draft labels before recording decisions. We fitted one cutoff by minimizing calibration classification errors. Ties favor the cutoff closest to zero, then the smaller cutoff. We froze the result, **-1.6875**, before measuring the benchmark. The base model's weights were fixed throughout.

## Benchmark and results

The source was XSTest at Git revision `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`. We excluded four previously flagged near-overlap candidates (IDs 85, 148, 196 and 432). We chose two requests from each of the 18 published request types by sorting a fixed hash of the case ID. The 36 selected requests included 20 labeled safe and 16 labeled unsafe. AuditKIT ran the classifier callback serially, without cached results or retries. Both thresholds use the same observed score on each case.

| Benchmark outcome | Original cutoff (0) | Calibrated cutoff (-1.6875) |
| --- | ---: | ---: |
| Unsafe requests correctly flagged, of 16 | 13 | 16 |
| Unsafe requests missed, of 16 | 3 | 0 |
| Benign requests correctly cleared, of 20 | 20 | 7 |
| Benign requests incorrectly flagged, of 20 | 0 | 13 |
| All requests correctly classified, of 36 | 33 | 23 |

Lowering the cutoff caught three previously missed unsafe requests and incorrectly flagged thirteen previously cleared benign requests. Accuracy fell from 91.7% to 63.9% on this sample. The original cutoff had higher accuracy here. The public data include all 36 decisions, all 24 calibration margins, the exact split and the learned cutoff.

## Scope and limitations

A classification decision tells us which requests the guard would flag. It does not measure what Tiny Aya would say in an open-ended conversation. Selecting two cases per type makes the coverage explicit but leaves each type sparsely sampled. The calibration set is small, its original labels were visible to the reviewer, and further overlap in meaning may remain. SafeTune effects on model responses and math-task utility remain unmeasured. The earlier six synthetic callback outputs are a separate software test.

## Reproducibility and attribution

The GitHub repository contains `classifier.py`, the exact study runner and an independent score checker. Download the score files into one directory and run `python verify_results.py /path/to/scores` from the code repository. This recomputes the threshold, checks test coverage and validates the confusion counts. Running fresh predictions requires your own authorized local Tiny Aya snapshot at revision `af89d219b53ed9b13b8a4645f9c8028973510324`. The classifier card gives the inference example. The published model files contain the learned decision head only. The run used Python 3.12, Torch 2.10 CPU, Transformers 5.22.0 and AuditKIT 1.1.9.

[Code](https://github.com/Jmbutcher1130/JimButcherStudio) · [Classifier](https://huggingface.co/jmbutcher1130/lexsi-tiny-aya-safety-classifier) · [Scores and split](https://huggingface.co/datasets/jmbutcher1130/lexsi-tiny-aya-safety-scores) · [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) · [Cohere Labs Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)

XSTest was created by Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi and Dirk Hovy. Its prompts are licensed under CC BY 4.0. The score release references the original IDs and omits the prompt text. [XSTest source and license](https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d)
