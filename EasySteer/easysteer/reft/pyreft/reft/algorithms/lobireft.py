import torch
from collections import OrderedDict

from ...core.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from .utils import LowRankRotateLayer


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        state_dict["learned_source"] = self.learned_source.data
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        if "learned_source" in state_dict:
            self.learned_source.data = state_dict["learned_source"].to(self.learned_source.device)

        # Recreate rotate_layer and load back the columns
        if "rotate_layer" in state_dict:
            overload_w = state_dict["rotate_layer"].to(self.learned_source.device)
            overload_w_width = overload_w.shape[-1]
            rotate_layer = LowRankRotateLayer(
                self.embed_dim, overload_w_width, init_orth=True).to(self.learned_source.device)
            self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
            assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        
        return 