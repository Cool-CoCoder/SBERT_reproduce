import torch
from torch import Tensor
from torch import nn
from typing import Dict


class Pooling(nn.Module):
    def __init__(self, pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True):
        super(Pooling, self).__init__()
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens

    def forward(self, features: Dict[str, Tensor]):
        output = None

        token_embeddings = features['embeddings']
        # 其实mask不是那么必要，暂时可以搁置
        attention_mask = features['attention_mask']

        if self.pooling_mode_mean_tokens:

            output = torch.mean(token_embeddings, 1)
        if self.pooling_mode_cls_token:
            raise NotImplementedError
        if self.pooling_mode_max_tokens:
            raise NotImplementedError

        return output
