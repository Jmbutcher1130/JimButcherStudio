import hashlib
import json
import auditkit as ak
from paired_bridge import paired_predictions

class SyntheticTreatment:
    active = False
    def install(self):
        self.active = True
    def remove(self):
        self.active = False

treatment = SyntheticTreatment()
cases = [dict(case_id=f"synthetic-core-{i}", prompt=f"software interface case {word}")
         for i, word in enumerate(("one", "two", "three"), 1)]

def generate(prompts):
    condition = "after" if treatment.active else "before"
    return [f"{condition}::{prompt}" for prompt in prompts]

manifest = {"purpose": "actual AuditKIT core only;synthetic generator/treatment",
            "cases": cases, "version": ak.__version__}
digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
result = paired_predictions(ak, cases, generate, treatment, digest,
                            lambda: not treatment.active, lambda: treatment.active)
expected = [{"case_id": c["case_id"], "condition": k, "output": k+"::"+c["prompt"]}
            for k in ("before", "after") for c in cases]
assert result["outputs"] == expected and not treatment.active
for row in result["outputs"]:
    print(json.dumps(row))
