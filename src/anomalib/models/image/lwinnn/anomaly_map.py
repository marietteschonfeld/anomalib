import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

class AnomalyMapGenerator(nn.Module):
    def __init__(self, window_size: int = 5) -> None:
        super().__init__()

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]), mode="bilinear", align_corners=False)
        return anomaly_map

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        return self.compute_anomaly_map(patch_scores, image_size)