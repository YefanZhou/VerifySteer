"""
Difference of Means Extractor
正负样本均值差方法
"""

import numpy as np
from tqdm.auto import tqdm
from .utils import StatisticalControlVector, extract_token_hiddens


class DiffMeanExtractor:
    """Difference of means method for control vector extraction"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """
        Extract control vectors using difference of means method.
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            normalize: 是否归一化向量
            token_pos: token位置，-1表示最后一个token（默认），支持int/"first"/"last"/"mean"/"max"/"min"
        """
        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )
        
        directions = {}
        
        for layer in tqdm(positive_hiddens.keys(), desc="Computing DiffMean directions"):
            # 计算正负样本的均值差
            mean_positive = np.mean(positive_hiddens[layer], axis=0)
            mean_negative = np.mean(negative_hiddens[layer], axis=0)
            direction = mean_positive - mean_negative
            
            if normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
            
            directions[layer] = direction.astype(np.float32)
        
        metadata = {
            "normalize": normalize,
            "token_pos": token_pos,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices) if negative_indices else len(all_hidden_states) - len(positive_indices)
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="diffmean",
            directions=directions,
            metadata=metadata
        ) 