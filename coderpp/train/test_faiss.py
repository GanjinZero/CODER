import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import faiss
import random
import string
import time
import pickle
import gc

batch_size = 64
device = torch.device("cuda:0")


def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                    torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    torch.cuda.empty_cache()
    return np.concatenate(output_list, axis=0)

def get_KNN(embeddings, k, use_multi_gpu=True):
    if use_multi_gpu:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(embeddings)
    else:
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embeddings)
    print(gpu_index.ntotal)
    similarity, indices = gpu_index.search(embeddings.astype(np.float32), k)
    del gpu_index
    gc.collect()
    return similarity, indices

def find_new_index(new_CODER_path, output_path, similarity_path, idx2string_path='data/idx2string.pkl', ori_CODER_path='GanjinZero/UMLSBert_ENG', use_multi_gpu=True):
    print('start finding new index...')
    config = AutoConfig.from_pretrained(ori_CODER_path)
    tokenizer = AutoTokenizer.from_pretrained(ori_CODER_path)
    print('start loading phrases...')
    with open('data/idx2string.pkl', 'rb') as f:
        phrase_list = list(pickle.load(f).values())
    print('done loading phrases')
    model = torch.load(new_CODER_path).to(device)
    embeddings = get_bert_embed(phrase_list, model, tokenizer, summary_method="MEAN", tqdm_bar=True)
    del model
    torch.cuda.empty_cache()
    print('start knn')
    similarity, indices = get_KNN(embeddings, 30, use_multi_gpu)  
    # similarity = np.zeros((len(phrase_list), 30))
    # indices = similarity
    with open(output_path, 'wb') as f:
        np.save(f, indices)
    with open(similarity_path, 'wb') as f:
        np.save(f, similarity)
    print('done knn')
    return None


if __name__ == "__main__":
    filename = "GanjinZero/UMLSBert_ENG"
    config = AutoConfig.from_pretrained(filename)
    tokenizer = AutoTokenizer.from_pretrained(filename)
    print('start loading phrases...')
    with open('data/idx2string.pkl', 'rb') as f:
        phrase_list = list(pickle.load(f).values())
    print('done loading phrases')
    # model = AutoModel.from_pretrained(
    #     filename,
    #     config=config).to(device)
    model = torch.load('output_testttt/last_model.pth').to(device)
    start = time.time()
    print('start testing...')
    embeddings = get_bert_embed(phrase_list[:100], model, tokenizer, summary_method="MEAN", tqdm_bar=True)
    print(embeddings.shape)
    # with open('data/embeddings.npy', 'wb') as f:
    #     np.save(f, embeddings)
    # print('done testing')
    # del model
    # torch.cuda.empty_cache()
    # embeddings = np.load('data/embeddings.npy')
    # print('start knn...')
    # similarity, indices = get_KNN(embeddings, 30, use_multi_gpu=True)  
    # with open('data/similarity.npy', 'wb') as f:
    #     np.save(f, similarity)
    # with open('data/indices.npy', 'wb') as f:
    #     np.save(f, indices)
    # print('done knn')
    # end = time.time()
    # print(end - start, 's')