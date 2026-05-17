import torch
import transformers
# from pyreft import *
import easysteer.reft.pyreft as pyreft
import os
# 导入必要的基类
from easysteer.reft.pyreft.core.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention
)


# 新的偏置干预类
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
        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))

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
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda"

# Step 1: loading the raw LM
prompt_no_input_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

model_name_or_path = "/data/zju-46/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token
# Step 7: ReFT model loading
reft_model_loaded = pyreft.ReftModel.load(
    "./results/ssv", model
)
reft_model_loaded.set_device(device) # move to device for inference

instruction = "Who are you?"
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)
base_unit_location = prompt["input_ids"].shape[-1] - 1
_, reft_response = reft_model_loaded.generate(
    prompt,
    unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True,
    max_new_tokens=512,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))