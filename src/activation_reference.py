from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ActivationDecision:
    status: str
    projection: float
    threshold: float
    refusal_mean: float
    compliance_mean: float
    margin: float


class ActivationReferenceClassifier:
    """
    Classifies new activations by comparing their projection to Stage 1
    refusal/compliance activation distributions.
    """

    def __init__(self, direction_vec: torch.Tensor, layer_idx: int, activations_data: dict):
        self.direction_vec = direction_vec.detach().float().cpu()
        self.layer_idx = layer_idx

        if layer_idx not in activations_data["activations"]:
            raise KeyError(f"Layer L{layer_idx} not found in Stage 1 activations.")

        acts = activations_data["activations"][layer_idx].float().cpu()
        labels = activations_data["labels"].cpu()
        projections = acts @ self.direction_vec

        refusal_proj = projections[labels == 0]
        compliance_proj = projections[labels == 1]
        if len(refusal_proj) == 0 or len(compliance_proj) == 0:
            raise ValueError("Stage 1 activations must contain both refusal and compliance labels.")

        self.refusal_mean = float(refusal_proj.mean().item())
        self.compliance_mean = float(compliance_proj.mean().item())
        self.refusal_std = float(refusal_proj.std(unbiased=False).item())
        self.compliance_std = float(compliance_proj.std(unbiased=False).item())
        self.threshold = (self.refusal_mean + self.compliance_mean) / 2.0
        self.compliance_is_higher = self.compliance_mean >= self.refusal_mean

    @classmethod
    def from_path(cls, activations_path: str | Path, direction_vec: torch.Tensor, layer_idx: int):
        data = torch.load(str(activations_path), map_location="cpu", weights_only=False)
        return cls(direction_vec=direction_vec, layer_idx=layer_idx, activations_data=data)

    def classify_projection(self, projection: float) -> ActivationDecision:
        if self.compliance_is_higher:
            is_compliance = projection >= self.threshold
            margin = projection - self.threshold
        else:
            is_compliance = projection <= self.threshold
            margin = self.threshold - projection

        return ActivationDecision(
            status="Compliance" if is_compliance else "Refusal",
            projection=float(projection),
            threshold=self.threshold,
            refusal_mean=self.refusal_mean,
            compliance_mean=self.compliance_mean,
            margin=float(margin),
        )

    def classify_activation(self, activation_vec: torch.Tensor) -> ActivationDecision:
        projection = torch.dot(
            activation_vec.detach().float().cpu(),
            self.direction_vec,
        ).item()
        return self.classify_projection(projection)
