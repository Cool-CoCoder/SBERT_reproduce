import torch
import torch.nn as nn
from transformers import RobertaModel
from Pooling import Pooling



class RobertaLi(nn.Module):
    def __init__(self, device, n_labels, embedding_size=768, concat_mode='no'):
        super().__init__()

        # 加载RoBERTa模型和分词器
        self.device = device
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.cel = nn.CrossEntropyLoss()
        self.num_concat = 2
        self.embedding_size = embedding_size
        self.pool = Pooling()

        self.concat_mode = concat_mode

        mode_list = ['abs', 'mul', 'both', 'no']
        # no则不做改变
        assert concat_mode in mode_list, 'concat_mode is not in the mode_list {}'.format(mode_list)
        if concat_mode == 'abs' or concat_mode == 'mul':
            self.num_concat = 3
        elif concat_mode == 'both':
            self.num_concat = 4
        # concat dim = 1
        self.classifier = nn.Linear(embedding_size * self.num_concat, n_labels)

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2):
        # RoBERTa编码
        out1 = self.roberta(input_ids_1, attention_mask=attention_mask_1)
        h1 = out1.last_hidden_state

        out2 = self.roberta(input_ids_2, attention_mask=attention_mask_2)
        h2 = out2.last_hidden_state
        
        
        f1 = {'embeddings': h1, 'attention_mask': attention_mask_1}
        f2 = {'embeddings': h2, 'attention_mask': attention_mask_2}
        p1 = self.pool(f1)
        p2 = self.pool(f2)
        # 这里需要加上pooling操作

        vector_concat = None
        concat_mode = self.concat_mode
        if concat_mode == 'abs':
            vector_concat = torch.abs(p1 - p2)
        elif concat_mode == 'mul':
            vector_concat = torch.abs(p1 * p2)
        elif concat_mode == 'both':
            vector_concat = torch.cat((torch.abs(p1 - p2), torch.abs(p1 * p2)), 1)

        if vector_concat:
            features = torch.cat([p1, p2, vector_concat], 1)
        else:
            features = torch.cat((p1, p2), 1)
            
        output = self.classifier(features)
        return output

    # 使用不同的concat方法
    def get_loss(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, y):

        loss = self.cel(self(input_ids_1, input_ids_2, attention_mask_1, attention_mask_2), y.to(self.device))
        return loss
