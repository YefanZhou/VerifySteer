import torch
from collections import OrderedDict
from transformers.activations import ACT2FN

from ...core.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        # Save proj_layer parameters
        for k, v in self.proj_layer.state_dict().items():
            state_dict[f"proj_layer.{k}"] = v
        # Save learned_source parameters
        for k, v in self.learned_source.state_dict().items():
            state_dict[f"learned_source.{k}"] = v
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        # Filter and load proj_layer parameters
        proj_layer_state = {k.replace("proj_layer.", ""): v for k, v in state_dict.items() 
                           if k.startswith("proj_layer.")}
        if proj_layer_state:
            self.proj_layer.load_state_dict(proj_layer_state)
        
        # Filter and load learned_source parameters
        learned_source_state = {k.replace("learned_source.", ""): v for k, v in state_dict.items() 
                               if k.startswith("learned_source.")}
        if learned_source_state:
            self.learned_source.load_state_dict(learned_source_state)
        
        return 