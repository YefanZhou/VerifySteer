import torch
from collections import OrderedDict

from ...core.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)


class BiasIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention
):
    """
    简单的偏置干预: BiasIntervention(h) = h + b
    只在隐藏状态上添加一个可学习的偏置向量
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # 创建一个可学习的偏置参数，维度与嵌入维度相同
        self.bias = torch.nn.Parameter(
            torch.zeros(self.embed_dim), requires_grad=True
        )
        # 添加dropout层用于正则化
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(self, base, source=None, subspaces=None):
        """
        前向传播：简单地将偏置加到输入上
        """
        # h + b
        output = base + self.bias
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        保存状态字典
        """
        state_dict = OrderedDict()
        state_dict["bias"] = self.bias.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        加载状态字典
        """
        if "bias" in state_dict:
            self.bias.data = state_dict["bias"].to(self.bias.device) 