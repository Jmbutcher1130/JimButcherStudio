"""Recompute the learned threshold and every reported count from public scores."""
import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from run_check import call_threshold, fit_threshold


def verify(root):
    cal = json.loads((root / "calibration-scores.json").read_text())
    scores = [json.loads(row) for row in (root / "scores.jsonl").read_text().splitlines()]
    expected = json.loads((root / "split.json").read_text())
    result = json.loads((root / "results.json").read_text())
    config = json.loads((root / "classifier_config.json").read_text())
    assert len(cal) == len({r["case_id"] for r in cal}) == 24
    assert len(scores) == len({r["case_id"] for r in scores}) == 36
    assert set(r["case_id"] for r in cal).isdisjoint(r["case_id"] for r in scores)
    assert {(r["case_id"],r["group"]) for r in scores} == {(r["case_id"],r["group"]) for r in expected}
    assert len(expected) == 36 and all(isfinite(r["margin"]) for r in cal + scores)
    assert Counter(r["gold"] for r in scores) == {"safe":20,"unsafe":16}
    threshold = fit_threshold([r["gold"] for r in cal], [r["margin"] for r in cal])
    assert threshold == config["unsafe_threshold"] == result["threshold"]
    assert result["test_count"] == len(scores) and result["calibration_count"] == len(cal)
    for method, t in (("baseline", 0.), ("calibrated", threshold)):
        counts = Counter()
        for r in scores:
            prediction = call_threshold(r["margin"], t)
            assert prediction == r[method]
            counts[r["gold"], prediction] += 1
        for g in ("safe", "unsafe"):
            for pred in ("safe", "unsafe"):
                assert result[method][g + "_as_" + pred] == counts[g, pred]
        assert result[method]["correct"] == counts["unsafe","unsafe"] + counts["safe","safe"]
    assert result["changed_from_wrong_to_correct"] == sum(r["baseline"] != r["gold"] == r["calibrated"] for r in scores)
    assert result["changed_from_correct_to_wrong"] == sum(r["baseline"] == r["gold"] != r["calibrated"] for r in scores)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_directory", type=Path)
    print(json.dumps(verify(parser.parse_args().scores_directory), indent=2))
