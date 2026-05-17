"""
PYREFT: Representation Finetuning for Language Models

A unified framework for representation editing and intervention methods.
"""

__version__ = "0.2.0"

# Core framework exports - Base classes
from .core import IntervenableModel, IntervenableConfig, build_intervenable_model

# Core interventions
from .core import (
    TrainableIntervention, 
    ConstantSourceIntervention,
    SourcelessIntervention,
    DistributedRepresentationIntervention,
    AdditionIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention
)

# Model creation utilities  
from .core import create_gpt2, create_gpt2_lm, create_llama

# REFT-specific exports
from .reft import (
    ReftModel,
    ReftConfig,
    get_reft_model,
    LoreftIntervention,
    NoreftIntervention,
    ReftTrainerForCausalLM
)

# Data processing exports
from .data import (
    ReftDataset, 
    ReftDataCollator,
    make_last_position_supervised_data_module
)

__all__ = [
    'IntervenableModel',
    'IntervenableConfig', 
    'build_intervenable_model',
    'TrainableIntervention',
    'ConstantSourceIntervention',
    'SourcelessIntervention', 
    'DistributedRepresentationIntervention',
    'AdditionIntervention',
    'RotatedSpaceIntervention',
    'LowRankRotatedSpaceIntervention',
    'create_gpt2',
    'create_gpt2_lm', 
    'create_llama',
    'ReftModel',
    'ReftConfig',
    'get_reft_model',
    'LoreftIntervention',
    'NoreftIntervention',
    'ReftTrainerForCausalLM',
    'ReftDataset',
    'ReftDataCollator',
    'make_last_position_supervised_data_module'
] 