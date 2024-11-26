import logging

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin

from .torch_model import SPALWinNNModel

logger = logging.getLogger(__name__)

__all__ = ["SPALWinNN"]


class SPALWinNN(MemoryBankMixin, AnomalyModule):

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],
        pre_trained: bool = True,
        interpolation_mode: str = "nearest",
        K_im: int = 50,
        anomaly_map_detection: bool = False,
        window_size:int = 1,
        pooling: bool = True,
    ) -> None:
        super().__init__()

        self.model: SPALWinNNModel = SPALWinNNModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            interpolation_mode=interpolation_mode,
            K_im=K_im,
            anomaly_map_detection=anomaly_map_detection,
            window_size=window_size,
            pooling=pooling,
        )

        self.stats: list[torch.Tensor] = []
        self.embeddings: list[torch.Tensor] = []
        self.samples=0

    @staticmethod
    def configure_optimizers() -> None:
        return

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        del args, kwargs 
        if self.samples <= 350:
            batch_embedding = self.model(batch["image"])
            self.embeddings.append(batch_embedding)
            self.samples += batch_embedding.shape[0]

    def fit(self) -> None:
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Generating memory bank")
        self.model.generate_memory_bank(embeddings)


    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.
        output = self.model(batch["image"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        return {"max_epochs": 1, "val_check_interval": 1.0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
