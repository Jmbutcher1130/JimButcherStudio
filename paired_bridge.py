"""Candidate AuditKIT/SafeTune paired bridge; inject existing objects."""
def paired_predictions(ak, cases, generate, wrapper, run_digest, verify_inactive, verify_active):
    """cases: [{case_id, prompt}]; generate: list[str] -> list[str]."""
    if not isinstance(run_digest, str) or len(run_digest) != 64 or any(c not in "0123456789abcdef" for c in run_digest):
        raise ValueError("run_digest must be a lowercase SHA-256 manifest digest")
    if not cases:
        raise ValueError("cases must not be empty")
    lookup = {}
    ids = set()
    for case in cases:
        case_id, prompt = case.get("case_id"), case.get("prompt")
        if not isinstance(case_id, str) or not case_id.strip() or not isinstance(prompt, str) or not prompt:
            raise ValueError("case id and prompt must be nonempty strings")
        if case_id in ids or prompt in lookup:
            raise ValueError("case ids and prompts must be unique")
        ids.add(case_id)
        lookup[prompt] = case_id


    def run(condition):
        seen = {}
        def capture(prompts):
            if not isinstance(prompts, list) or any(p not in lookup or p in seen for p in prompts):
                raise ValueError("unexpected or repeated prompt")
            if len(prompts) != len(set(prompts)):
                raise ValueError("repeated prompt in batch")
            outputs = generate(prompts)
            if not isinstance(outputs, list) or len(outputs) != len(prompts) or any(not isinstance(x, str) for x in outputs):
                raise ValueError("invalid model output batch")
            for prompt, output in zip(prompts, outputs):
                seen[prompt] = output
            return outputs
        dataset = [ak.Sample(input=c["prompt"]) for c in cases]
        model = ak.model.CallableModel(capture, name="lexsi:" + run_digest + ":" + condition)
        result = ak.evaluate(dataset, model=model, config=ak.RunConfig(max_retries=0, concurrency=1), cache=False)
        if set(seen) != set(lookup):
            raise RuntimeError("evaluation did not produce exactly one output per case")
        return result, [{"case_id":lookup[c["prompt"]], "condition":condition,
                         "output":seen[c["prompt"]]} for c in cases]


    if not callable(verify_inactive) or not callable(verify_active) or verify_inactive() is not True:
        raise RuntimeError("baseline requires verified inactive treatment")
    before_result, before = run("before")
    try:
        wrapper.install()
        if verify_active() is not True:
            raise RuntimeError("treatment activation not verified")
        after_result, after = run("after")
    finally:
        wrapper.remove()
        if verify_inactive() is not True:
            raise RuntimeError("treatment cleanup not verified")
    return {"before_audit":before_result, "after_audit":after_result,
            "outputs":before + after}
