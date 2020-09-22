import torch
from torch import nn
import os


class LinearModel(nn.Module):
    def __init__(self, label_count, embedding_type, embedding, freeze_embedding=True):
        super(LinearModel, self).__init__()
        self.embedding_type = embedding_type
        if self.embedding_type in ["word", "cui"]:
            self.embedding = nn.Embedding.from_pretrained(embedding)
            self.input_dim = self.embedding.weight.shape[1]
            if freeze_embedding:
                self.embedding.weight.required_grad = False
        if self.embedding_type == "bert":
            self.embedding = embedding
            self.input_dim = 768
            if freeze_embedding:
                for name, param in self.embedding.named_parameters():
                    param.requires_grad = False
        self.linear = nn.Linear(self.input_dim * 2, label_count)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x0, x1, length_0=None, length_1=None, label=None):
        count = x0.shape[0]
        x = torch.cat((x0, x1), dim=0)
        emb = self.embedding(x)

        #print(x.shape, emb.shape, length_0.shape)

        if self.embedding_type == "word":
            emb = torch.sum(emb, dim=1)
            length = torch.cat((length_0, length_1)).reshape(-1, 1).expand_as(emb)
            emb = emb / length
        if self.embedding_type == "cui":
            pass
        if self.embedding_type == "bert":
            emb = emb[1]

        emb_0 = emb[0:count]
        emb_1 = emb[count:]
        feature = torch.cat((emb_0, emb_1), dim=1)
        pred = self.linear(feature)

        if label is not None:
            loss = self.loss_fn(pred, label)
            return pred, loss
        return pred, 0.
