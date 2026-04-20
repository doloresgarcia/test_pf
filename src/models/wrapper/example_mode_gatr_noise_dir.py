import torch

# from src.models.gravnet_3_tracking import GravnetModel
from src.models.GATr.Gatr_pf_e_noise_knn import ExampleWrapper as GravnetModel

# from src.models.point_transformer.point_transformer import (
#     PointTransformerOC as GravnetModel,
# )


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs) -> None:
        super().__init__()
        self.mod = GravnetModel(args, dev, **kwargs)

    def forward(self, g, step_count):
        return self.mod(g, step_count)


def get_model(data_config, args, dev, **kwargs):
    model = GraphTransformerNetWrapper(args, dev, **kwargs)

    model_info = {
    }

    return model, model_info


def get_loss(data_config, **kwargs):

    return torch.nn.MSELoss()
