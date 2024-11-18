from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

from src.anomalib import TaskType
from src.anomalib.data import Folder
from src.anomalib.data.utils import read_image
from src.anomalib.engine import Engine
from src.anomalib.models import LWinNN
import torch
from anomalib.data import MVTec


category = 'pill'

# Create the datamodule
datamodule = MVTec(num_workers=0,category=category)
datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
datamodule.setup()

# datamodule = MVTec(num_workers=0)
# model = Patchcore(backbone="resnet18")
model = LWinNN(backbone="resnet18")

# # start training
engine = Engine(task=TaskType.SEGMENTATION)
engine.fit(model=model, datamodule=datamodule)

# # load best model from checkpoint before evaluating
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)