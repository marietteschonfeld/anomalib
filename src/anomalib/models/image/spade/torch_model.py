from random import sample
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import MultiVariateGaussian, TimmFeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .anomaly_map import AnomalyMapGenerator
from math import floor

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler

# defaults from the paper
# _N_FEATURES_DEFAULTS = {
#     "resnet18": 100,
#     "wide_resnet50_2": 550,
# }


# def _deduce_dims(
#     feature_extractor: TimmFeatureExtractor,
#     input_size: tuple[int, int],
#     layers: list[str],
# ) -> tuple[int, int]:
#     """Run a dry run to deduce the dimensions of the extracted features.

#     Important: `layers` is assumed to be ordered and the first (layers[0])
#                 is assumed to be the layer with largest resolution.

#     Returns:
#         tuple[int, int]: Dimensions of the extracted features: (n_dims_original, n_patches)
#     """
#     dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

#     # the first layer in `layers` has the largest resolution
#     first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
#     n_patches = torch.tensor(first_layer_resolution).prod().int().item()

#     # the original embedding size is the sum of the channels of all layers
#     n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore[misc]

#     return n_features_original, n_patches


class SPADEModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        K_im: int = 50,
        K_pix: int = 1,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=pre_trained,
        ).eval()

        self.K_im = K_im
        self.K_pix = K_pix

        self.register_buffer("memory_bank", torch.Tensor())

        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator()


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            pred_scores, top_K_images = self.compute_anomaly_scores(embeddings)
            patch_scores = self.compute_patch_scores(embeddings, top_K_images)

            anomaly_map = self.anomaly_map_generator(patch_scores, output_size)
            output = {"pred_score": pred_scores, 
                        "anomaly_map": anomaly_map}

        return output

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings
    
    def generate_memory_bank(self, embeddings):
        N,C,H,W = embeddings.shape
        self.memory_bank = embeddings.permute(0,2,3,1).reshape(N, H*W, C)

    
    @staticmethod
    def nearest_neighbors(embedding, bank):
        distances = torch.cdist(embedding, bank, compute_mode='use_mm_for_euclid_dist')
        patch_scores, _ = distances.min(dim=-1)
        return patch_scores

    def compute_patch_scores(self, embedding: torch.Tensor, top_K_images: torch.Tensor) -> torch.Tensor:
        batch_size, embedding_size, H, W = embedding.shape
        embedding = embedding.permute(0,2,3,1).reshape(batch_size, H*W, embedding_size)
        dists = []

        for i in range(batch_size):
            temp_embedding = embedding[i].unsqueeze(1)
            temp_bank = self.memory_bank[top_K_images[i]].transpose(0,1)
            dist = self.nearest_neighbors(temp_embedding, temp_bank).reshape(1,H,W)
            dists.append(dist)
        patch_scores = torch.stack(dists)
        return patch_scores
    
    def compute_anomaly_scores(self, embedding: torch.Tensor) -> torch.Tensor:
        batch_size, train_size = embedding.shape[0], self.memory_bank.shape[0]
        embedding = embedding.permute(0,2,3,1)
        distances = torch.cdist(embedding.reshape(batch_size,-1), self.memory_bank.reshape(train_size, -1), compute_mode='use_mm_for_euclid_dist')
        pred_scores, top_K_images = torch.topk(distances, k=self.K_im, dim=1, largest=False)
        return pred_scores.mean(dim=1), top_K_images
    