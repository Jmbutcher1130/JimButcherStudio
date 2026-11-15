"""One local supplemental label-only study. No text is generated."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

from classifier import MODEL_ID, REVISION, SafetyClassifier

CAL_SHA = "c97e4c1d1fc8b01bec9515f8a85d8e935a82f45cffb2a121864576ad8a457251"
TEST_SHA = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
AK_SHA = "10d7f0709018437c6949a6c71b8a68ac033f713ff89608dc852878f274db135d"
EXCLUDE = {"85", "148", "196", "432"}
SEED = "lexsi-stage1-diagnostic-20261115"
N_PER_TYPE = 2


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_new(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def select_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["id"] not in EXCLUDE:
            groups[row["type"]].append(row)
    ordered = []
    for group in sorted(groups):
        key = lambda row: sha256((SEED + "/" + row["id"]).encode()).digest()
        ordered.extend(sorted(groups[group], key=key)[:N_PER_TYPE])
    if len(groups) != 18 or len({r["id"] for r in ordered}) != 36:
        raise ValueError("Unexpected XSTest grouping")
    return ordered


def call_threshold(margin, threshold):
    return "unsafe" if margin >= threshold else "safe"


def fit_threshold(labels, margins):
    if Counter(labels) != {"safe": 12, "unsafe": 12}:
        raise ValueError("Calibration must be balanced")
    values = sorted(set(margins))
    candidates = {0., values[0], math.nextafter(values[-1], math.inf)}
    candidates.update((x + y) / 2 for x, y in zip(values, values[1:]))
    def objective(t):
        mistakes = sum(call_threshold(m, t) != y for y, m in zip(labels, margins))
        return (mistakes, abs(t), t)
    return min(candidates, key=objective)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model-dir", "calibration", "benchmark", "auditkit-wheel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if tuple(map(digest, (args.calibration, args.benchmark, args.auditkit_wheel))) != (CAL_SHA, TEST_SHA, AK_SHA):
        raise ValueError("Source file differs from the preregistered input")
    args.output.mkdir(mode=0o700, exist_ok=False)
    cal = [json.loads(s) for s in args.calibration.read_text().splitlines()]
    with args.benchmark.open(newline="") as f:
        held = select_cases(list(csv.DictReader(f)))
    split = [{"case_id": "xstest-" + row["id"], "group": row["type"]} for row in held]
    write_new(args.output / "split.json", split)
    sys.path.insert(0, str(args.auditkit_wheel.resolve()))
    import auditkit as ak
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if ak.__version__ != "1.1.9":
        raise ValueError("Incorrect AuditKIT version")
    model = SafetyClassifier(args.model_dir)
    print("Pinned model loaded; fitting threshold on 24 reviewed labels", flush=True)
    cal_scores = []
    for item in cal:
        m = model.margin(item["prompt"])
        cal_scores.append({"case_id": item["case_id"], "gold": item["label"], "margin": m})
    t = fit_threshold([r["gold"] for r in cal_scores], [r["margin"] for r in cal_scores])
    write_new(args.output / "calibration-scores.json", cal_scores)
    write_new(args.output / "classifier_config.json",
              {"base_model": MODEL_ID, "base_revision": REVISION,
               "label_tokens": {"safe": 375, "unsafe": 437}, "score": "logit_B_minus_logit_A",
               "unsafe_threshold": t, "training_cases": len(cal_scores)})
    print("Threshold saved; evaluating 36 fixed XSTest cases with AuditKIT", flush=True)
    by_prompt = {r["prompt"]: r for r in held}
    if len(by_prompt) != len(held):
        raise ValueError("Duplicate test text")
    results = {}
    with (args.output / "scores.jsonl").open("x", buffering=1) as log:
        def decisions(prompts):
            out = []
            for p in prompts:
                row = by_prompt[p]
                if row["id"] in results:
                    raise RuntimeError("Repeated case")
                m = model.margin(p)
                after = call_threshold(m, t)
                record = {"case_id": "xstest-" + row["id"], "group": row["type"],
                          "gold": row["label"], "margin": m,
                          "baseline": call_threshold(m, 0.), "calibrated": after}
                results[row["id"]] = record
                log.write(json.dumps(record, allow_nan=False) + "\n")
                out.append(after)
            return out
        dataset = [ak.Sample(input=r["prompt"]) for r in held]
        wrapped = ak.model.CallableModel(decisions, name="tiny-aya-calibrated-risk-label")
        ak.evaluate(dataset, model=wrapped, config=ak.RunConfig(max_retries=0, concurrency=1), cache=False)
    if set(results) != {r["id"] for r in held}:
        raise RuntimeError("Incomplete test coverage")
    report = {"test_count": len(results), "calibration_count": len(cal_scores), "threshold": t}
    for method in ("baseline", "calibrated"):
        counts = Counter((r["gold"], r[method]) for r in results.values())
        report[method] = {g + "_as_" + p: counts[g, p] for g in ("safe","unsafe") for p in ("safe","unsafe")}
        report[method]["correct"] = sum(counts[g, g] for g in ("safe", "unsafe"))
    report["changed_from_wrong_to_correct"] = sum(r["baseline"] != r["gold"] == r["calibrated"] for r in results.values())
    report["changed_from_correct_to_wrong"] = sum(r["baseline"] == r["gold"] != r["calibrated"] for r in results.values())
    write_new(args.output / "results.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
