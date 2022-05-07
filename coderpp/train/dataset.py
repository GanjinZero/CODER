import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from load_umls import UMLS
from torch.utils.data import Dataset, DataLoader
from random import sample
from torch.utils.data.sampler import RandomSampler
# import ipdb
from time import time
import json
import pickle
import ahocorasick
import torch

class UMLSDataset(Dataset):
    def __init__(self, umls_folder='../umls', model_name_or_path='GanjinZero/UMLSBert_ENG', idx2phrase_path='data/idx2string.pkl', phrase2idx_path='data/string2idx.pkl', indices_path='data/indices.npy', max_length=32):
        super().__init__()
        self.umls = UMLS(umls_folder, phrase2idx_path=phrase2idx_path, idx2phrase_path=idx2phrase_path)
        self.indices = np.load(indices_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.cui2idx = {cui: idx for idx, cui in enumerate(self.umls.cui2stridx.keys())}
        self.idx2phrase = self._load_pickle(idx2phrase_path)
        self.max_length = max_length

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def tokenize_one(self, string):
        tokenized = self.tokenizer.encode_plus(string, 
                                               max_length=self.max_length, 
                                               truncation=True, 
                                               pad_to_max_length=True, 
                                               add_special_tokens=True)
        return tokenized['input_ids'], tokenized['attention_mask']


    def __getitem__(self, index):
        input_str_list = []     # [current_str, top30_str, 30*rand_same_cui_str]
        current_str_idx = self.umls.stridx_list[index]
        input_str_list.append(self.idx2phrase[current_str_idx])
        input_str_list = input_str_list + [self.idx2phrase[idx] for idx in self.indices[current_str_idx]]
        current_cui = self.umls.str2cui[self.idx2phrase[current_str_idx]]
        stridx_set_for_current_cui = self.umls.cui2stridx[current_cui]
        idx_list = sample(stridx_set_for_current_cui - {current_str_idx}, min(30, len(stridx_set_for_current_cui) - 1))
        if len(idx_list) < 30:
            idx_list += [current_str_idx] * (30 - len(idx_list))
        input_str_list += [self.idx2phrase[idx] for idx in idx_list]
        input_cui_idx_list = [self.cui2idx[self.umls.str2cui[s]] for s in input_str_list]
        input_ids = [self.tokenize_one(s)[0] for s in input_str_list]
        attention_mask = [self.tokenize_one(s)[1] for s in input_str_list]
        return input_ids, input_cui_idx_list, attention_mask
    
    def __len__(self):
        return len(self.umls.stridx_list)

def my_collate_fn(batch):
    output_ids = torch.LongTensor([sample[0] for sample in batch])
    output_label = torch.LongTensor([sample[1] for sample in batch])
    output_attention_mask = torch.LongTensor([sample[2] for sample in batch])
    output_ids = output_ids.reshape(output_ids.shape[0] * output_ids.shape[1], output_ids.shape[2])
    output_label = output_label.reshape(output_label.shape[0] * output_label.shape[1], )
    output_attention_mask = output_attention_mask.reshape(output_attention_mask.shape[0] * output_attention_mask.shape[1], output_attention_mask.shape[2])
    return output_ids, output_label, output_attention_mask

    
if __name__ == '__main__':
    umls_dataset = UMLSDataset()
    print(umls_dataset[400])
    print(len(umls_dataset[400][0]))
    umls_dataloader = DataLoader(umls_dataset,
                                 batch_size=5, 
                                 shuffle=True,
                                 num_workers=1, 
                                 pin_memory=True, 
                                 drop_last=True,
                                 collate_fn=my_collate_fn)
    data, label, mask = next(iter(umls_dataloader))
    print(data.shape)
    print(label.shape)
    print(mask.shape)