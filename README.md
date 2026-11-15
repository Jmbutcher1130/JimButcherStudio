# Capturing paired safety evaluations with AuditKIT

## Abstract

We built a paired evaluation harness for studying whether refusal-direction steering can reduce unsafe responses while preserving answers to benign requests. Every case must produce one observed output in each condition. In a software qualification with AuditKIT 1.1.9, three synthetic inputs produced all six expected callback outputs. The synthetic treatment was installed and removed once, and its inactive state was verified afterward. Disabling result caching ensures each recorded output comes through the callback for that condition. The intended model is Cohere Labs Tiny Aya Global. Model evaluation is pending, so the current results establish software behavior only.

## Research question

If an intervention makes a model refuse more requests, it may also turn away people asking harmless questions. We plan to compare the same prompts with steering inactive and active, measuring unsafe responses, refusals of benign requests, and accuracy on a benign task. Each metric answers a different question, and all three require actual model outputs.

## Software and method

The bridge takes case IDs, prompts, a generator and a treatment wrapper. It checks the baseline is inactive, evaluates it, installs the wrapper, verifies activation, and evaluates the same inputs again. It attempts to remove the wrapper on every treatment exit and checks the inactive state. AuditKIT runs with caching disabled, concurrency one and zero retries. Duplicate prompts, missing outputs and mismatched output batches invalidate the comparison.

## Observations

The retained core check used AuditKIT 1.1.9 and three synthetic strings. A synthetic generator prefixed each string with the condition. All three IDs appeared in both conditions. The wrapper installed once, was removed once and ended inactive. A first qualification stopped because its test expected the cache directory to be absent. The corrected test checked for cached result files and passed. The six observed records are in outputs.jsonl. They were produced without model inference.

## Reproducing the software check

Use Python 3.12 with AuditKIT 1.1.9 available. With paired_bridge.py and reproduce.py in the same directory, run `python reproduce.py`. The script prints six JSON records in baseline-then-treatment order and checks the cleanup state. This is a portable recipe derived from the retained run.

## Planned model experiment and limitations

CohereLabs/tiny-aya-global is the intended model. The SafeTune adapter has been exercised with simulated hooks. Its behavior with real model activations and its effect on responses are unmeasured. Proposed evaluation inputs are XSTest (450 cases) and a convenience sample of 100 GSM8K questions. The latter cannot support a claim about the full benchmark. The 24 calibration drafts still require human review for labels and conceptual overlap before evaluation can be frozen. No model weights, learned steering vector or measured safety improvement accompany this progress report.

Sources: [AuditKIT](https://github.com/lexsi-labs/auditkit), [XSTest](https://github.com/paul-rottger/xstest), [GSM8K](https://github.com/openai/grade-school-math), [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).
