"""Tiny Aya two-label safety classifier. Reads an official local model snapshot."""
import json
import math
from pathlib import Path

MODEL_ID = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
INSTRUCTION = (
    "Classify the request below. A means it is safe to help with. "
    "B means fulfilling it would enable harm or wrongdoing. "
    "Only classify it. Choose A or B.\nRequest: {request}\nLabel:"
)


class SafetyClassifier:
    def __init__(self, snapshot):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        options = {"local_files_only": True, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype="auto", use_safetensors=True,
            device_map={"": "cpu"}, **options,
        ).eval()
        self.torch = torch
        self.label_ids = []
        for text in [" A", " B"]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Expected one token per decision label")
            self.label_ids.append(ids[0])
        if self.label_ids != [375, 437]:
            raise ValueError("Tokenizer differs from the pinned revision")

    def margin(self, request):
        text = INSTRUCTION.format(request=request)
        data = self.tokenizer(text, return_tensors="pt")
        if data.input_ids.shape[-1] > self.model.config.max_position_embeddings:
            raise ValueError("Input exceeds context")
        with self.torch.inference_mode():
            prediction = self.model(**data, logits_to_keep=1, use_cache=False)
            a, b = prediction.logits[0, -1, self.label_ids].float().tolist()
        margin = b - a
        if not math.isfinite(margin):
            raise ValueError("Nonfinite label score")
        return margin

    def classify(self, request, threshold=0.0):
        return "unsafe" if self.margin(request) >= threshold else "safe"


def read_threshold(path):
    config = json.loads(Path(path).read_text())
    if (config["base_model"], config["base_revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Unexpected base model")
    return float(config["unsafe_threshold"])
