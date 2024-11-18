import logging

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin

from .torch_model import LWinNNModel

logger = logging.getLogger(__name__)

__all__ = ["LWinNN"]


class LWinNN(MemoryBankMixin, AnomalyModule):

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],
        pre_trained: bool = True,
        window_size:int = 5,
        pooling: bool = True,
    ) -> None:
        super().__init__()

        self.model: LWinNNModel = LWinNNModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            window_size=window_size,
            pooling=pooling,
        )

        self.stats: list[torch.Tensor] = []
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:
        return

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        del args, kwargs 

        batch_embedding = self.model(batch["image"])
        self.embeddings.append(batch_embedding)

    def fit(self) -> None:
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Generating memory bank")
        self.model.generate_memory_bank(embeddings)


    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])
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
