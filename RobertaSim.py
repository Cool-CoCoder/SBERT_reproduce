import torch
import torch.nn as nn
from transformers import RobertaModel
from Pooling import Pooling
from torch.nn import MSELoss

class RobertaSim(nn.Module):
    def __init__(self, device):
        super().__init__()

        # 加载RoBERTa模型和分词器
        self.device = device
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.pool = Pooling()
        self.mse = MSELoss()

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2):
        # RoBERTa编码
        out1 = self.roberta(input_ids_1, attention_mask=attention_mask_1)
        l_h1 = out1.last_hidden_state

        out2 = self.roberta(input_ids_2, attention_mask=attention_mask_2)
        l_h2 = out2.last_hidden_state
        return l_h1, l_h2

    def get_cosine_sim(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2):
        # pool操作
        h1, h2 = self(input_ids_1, input_ids_2, attention_mask_1, attention_mask_2)
        f1 = {'embeddings': h1, 'attention_mask': attention_mask_1}
        f2 = {'embeddings': h2, 'attention_mask': attention_mask_2}
        p1 = self.pool(f1)
        p2 = self.pool(f2)

        return F.cosine_similarity(p1, p2, dim=1, eps=1e-8)

    def get_loss(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, y):
        # print(input_ids_1,input_ids_2)
        sim = self.get_cosine_sim(input_ids_1.to(self.device), input_ids_2.to(self.device),
                                  attention_mask_1.to(self.device), attention_mask_2.to(self.device))
        loss = self.mse(sim, y.to(self.device))
        # print(sim,y,loss)
        return loss
