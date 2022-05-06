#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import AMSoftmax
from pytorch_metric_learning import losses, miners

class UMLSFinetuneModel(nn.Module):
    def __init__(self, device, model_name_or_path, cui_label_count, cui_loss_type="ms_loss"):
        super(UMLSFinetuneModel, self).__init__()

        self.device = device
        self.model_name_or_path = model_name_or_path
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(self.model_name_or_path, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.feature_dim = 768

        self.cui_loss_type = cui_loss_type
        self.cui_label_count = cui_label_count

        if self.cui_loss_type == "softmax":
            self.cui_loss_fn = nn.CrossEntropyLoss()
            self.linear = nn.Linear(self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "am_softmax":
            self.cui_loss_fn = AMSoftmax(self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "ms_loss":
            self.cui_loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50)
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    def softmax(self, logits, label):
        loss = self.cui_loss_fn(logits, label)
        return loss
    
    def am_softmax(self, pooled_output, label):
        loss, _ = self.cui_loss_fn(pooled_output, label)
        return loss
    
    def ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.cui_loss_fn(pooled_output, label, pairs)
        return loss
    
    def calculate_loss(self, pooled_output=None, logits=None, label=None):
        if self.cui_loss_type == "softmax":
            return self.softmax(logits, label)
        if self.cui_loss_type == "am_softmax":
            return self.am_softmax(pooled_output, label)
        if self.cui_loss_type == "ms_loss":
            return self.ms_loss(pooled_output, label)    
    
    def get_sentence_feature(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]
        return pooled_output

    def forward(self, input_ids, cui_label, attention_mask):
        pooled_output = self.get_sentence_feature(input_ids, attention_mask)
        if self.cui_loss_type == "softmax":
            logits = self.linear(pooled_output)
        else:
            logits = None
        cui_loss = self.calculate_loss(pooled_output, logits, cui_label)            
        loss = cui_loss
        return loss
