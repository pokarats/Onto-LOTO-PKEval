"""Local multimodal baselines and visual-evidence retrieval.

CLIP uses aligned embedding cosine. PaliGemma uses an explicitly different,
generative yes/no score: it must NOT be reported as the paper's embedding score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loto_pke.common import require_module
from loto_pke.llm import TextGenerator
from loto_pke.parsing import source_images
from loto_pke.schema import PublicDataset


@dataclass(frozen=True)
class ImageCandidate:
    image_id: str
    document_id: str
    path: Path


class ImageTextScorer(Protocol):
    method: str

    def score(self, image_path: Path, text: str) -> float:
        """Higher means better image/text correspondence."""


class ClipScorer:
    method = "clip_cosine"

    def __init__(
        self, model_name: str = "openai/clip-vit-large-patch14-336", offline: bool = True,
    ) -> None:
        self.torch = require_module("torch", "Install torch")
        transformers = require_module("transformers", "Install transformers")
        self.image_module = require_module("PIL.Image", "Install Pillow")
        self.model = transformers.CLIPModel.from_pretrained(model_name, local_files_only=offline)
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name, local_files_only=offline)
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.image_cache: dict[str, Any] = {}
        self.text_cache: dict[str, Any] = {}

    def _embed_image(self, image_path: Path) -> Any:
        key = str(image_path.resolve())
        if key not in self.image_cache:
            with self.image_module.open(image_path) as image:
                inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.no_grad():
                output = self.model.get_image_features(**inputs)
                tensor = output.pooler_output if hasattr(output, "pooler_output") else output
                normalized = self.torch.nn.functional.normalize(tensor, dim=-1)
            self.image_cache[key] = normalized[0].cpu()
        return self.image_cache[key]

    def _embed_text(self, text: str) -> Any:
        if text not in self.text_cache:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.no_grad():
                output = self.model.get_text_features(**inputs)
                tensor = output.pooler_output if hasattr(output, "pooler_output") else output
                normalized = self.torch.nn.functional.normalize(tensor, dim=-1)
            self.text_cache[text] = normalized[0].cpu()
        return self.text_cache[text]

    def score(self, image_path: Path, text: str) -> float:
        return float(self.torch.dot(self._embed_image(image_path), self._embed_text(text)).item())


class PaliGemmaScorer:
    method = "paligemma_yes_no_first_token_logit_margin"

    def __init__(
        self, model_name: str = "google/paligemma2-3b-mix-448", offline: bool = True,
    ) -> None:
        self.torch = require_module("torch", "Install torch")
        transformers = require_module("transformers", "Install transformers")
        self.image_module = require_module("PIL.Image", "Install Pillow")
        self.processor = transformers.AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        model_dtype = (
            self.torch.float16
            if self.torch.cuda.is_available() and self.torch.cuda.get_device_capability(0)[0] < 8
            else "auto"
        )
        self.model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, local_files_only=offline, device_map="auto", dtype=model_dtype,
        ).eval()
        tokenizer = self.processor.tokenizer
        self.yes_token = tokenizer.encode(" yes", add_special_tokens=False)[0]
        self.no_token = tokenizer.encode(" no", add_special_tokens=False)[0]

    def score(self, image_path: Path, text: str) -> float:
        prompt = (
            "Is this image visual evidence for the following documented LOTO action? "
            f"{text}\nAnswer yes or no:"
        )
        with self.image_module.open(image_path) as image:
            inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return float((logits[self.yes_token] - logits[self.no_token]).item())


class HFVisionGenerator(TextGenerator):
    """Image-text-to-text model using its own chat template and local weights."""

    def __init__(
        self, model_name: str,
        offline: bool = True,
        max_new_tokens: int = 1024,
        seed: int = 42,
        dtype: str = "auto",
    ) -> None:
        self.torch = require_module("torch", "Install torch and accelerate")
        transformers = require_module("transformers", "Install a Transformers release supporting the model")
        self.image_module = require_module("PIL.Image", "Install Pillow")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported model dtype: {dtype}")
        torch_dtype = "auto" if dtype == "auto" else getattr(self.torch, dtype)
        if dtype == "auto" and self.torch.cuda.is_available() and self.torch.cuda.get_device_capability(0)[0] < 8:
            torch_dtype = self.torch.float16
        self.processor = transformers.AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=offline, dtype=torch_dtype, device_map="auto",
            attn_implementation="sdpa",
        ).eval()

    def generate(self, system: str, prompt: str, images: list[Path] | None = None) -> str:
        content: list[dict[str, Any]] = []
        opened = []
        try:
            for image_path in images or []:
                image = self.image_module.open(image_path).convert("RGB")
                opened.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": system + "\n\n" + prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            with self.torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            return self.processor.decode(generated[0][input_len:], skip_special_tokens=True)
        finally:
            for image in opened:
                image.close()


def build_image_candidates(dataset: PublicDataset, data_root: Path) -> list[ImageCandidate]:
    root = data_root.resolve()
    if root.name != "public_data" and (root / "public_data").is_dir():
        root = root / "public_data"
    candidates = []
    for document in dataset.documents:
        if not document.selected:
            continue
        if document.source_kind == "image" and document.source:
            candidates.append(ImageCandidate(f"{document.document_id}:page1", document.document_id, root / document.source.path))
        else:
            for index, page in enumerate(document.rendered_pages, start=1):
                candidates.append(ImageCandidate(f"{document.document_id}:page{index}", document.document_id, root / page.path))
    return candidates


def rank_visual_evidence(
    queries: list[dict[str, str]], candidates: list[ImageCandidate],
    scorer: ImageTextScorer, top_k: int = 2,
) -> list[dict[str, Any]]:
    """Returns image rankings plus a separately labeled document-level proxy."""

    results = []
    for query in queries:
        scores = sorted(
            ((candidate, scorer.score(candidate.path, query["text"])) for candidate in candidates),
            key=lambda item: (-item[1], item[0].image_id),
        )
        ranked = [
            {"image_id": item.image_id, "document_id": item.document_id, "score": score}
            for item, score in scores[:top_k]
        ]
        results.append(
            {
                "step_id": query["step_id"],
                "document_id": query["document_id"],
                "method": scorer.method,
                "ranked_images": ranked,
                "document_proxy_hit_at_1": bool(scores and scores[0][0].document_id == query["document_id"]),
                "document_proxy_hit_at_2": any(item.document_id == query["document_id"] for item, _ in scores[:2]),
                "fine_grained_acc_at_k": None,
                "fine_grained_status": "missing_step_to_image_gold_labels",
            }
        )
    return results


def grounding_label_template(
    queries: list[dict[str, str]], candidates: list[ImageCandidate],
) -> list[dict[str, Any]]:
    """Produce blank human-review tasks, never pseudo-gold alignments."""

    available = [candidate.image_id for candidate in candidates]
    return [
        {
            "step_id": query["step_id"],
            "document_id": query["document_id"],
            "instruction": query["text"],
            "candidate_image_ids": available,
            "gold_image_ids": [],
            "gold_bbox": None,
            "annotation_status": "needs_human_annotation",
        }
        for query in queries
    ]


def grounding_accuracy(
    rankings: list[dict[str, Any]], gold: list[dict[str, Any]] | None,
    ks: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    if gold is None:
        return {"status": "not_evaluable", "reason": "missing_step_to_image_gold_labels"}
    indexed = {item["step_id"]: item for item in gold if item.get("annotation_status") == "verified"}
    hit_counts = {k: 0 for k in ks}
    evaluated = 0
    retrieved_gold_scores: list[float] = []
    for ranking in rankings:
        label = indexed.get(ranking["step_id"])
        if label is None or not label.get("gold_image_ids"):
            continue
        evaluated += 1
        for k in ks:
            top = {item["image_id"] for item in ranking.get("ranked_images", [])[:k]}
            hit_counts[k] += bool(top.intersection(label["gold_image_ids"]))
        retrieved_gold_scores.extend(
            float(item["score"])
            for item in ranking.get("ranked_images", [])
            if item["image_id"] in label["gold_image_ids"]
        )
    return {
        "status": "evaluated" if evaluated else "not_evaluable",
        "n_verified_queries": evaluated,
        "mean_retrieved_gold_score": (
            sum(retrieved_gold_scores) / len(retrieved_gold_scores)
            if retrieved_gold_scores else None
        ),
        "retrieved_gold_score_note": "Truncated top-k only; CLIP score is cosine, PaliGemma score is a logit margin",
        **{f"acc_at_{k}": hit_counts[k] / evaluated if evaluated else None for k in ks},
    }
