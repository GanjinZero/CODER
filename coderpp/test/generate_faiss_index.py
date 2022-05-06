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
import argparse

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

def get_KNN(embeddings, k):
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

def find_new_index(indices_path, similarity_path, embedding_path, phrase2idx_path, tokenizer_name, model_name_or_path):
    print('start finding new index...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if model_name_or_path[-4:] == '.pth':
        model = torch.load(model_name_or_path).to(device)
    else:
        model = AutoModel.from_pretrained(model_name_or_path).to(device)
    print('start loading phrases...')
    with open(phrase2idx_path, 'rb') as f:
        phrase2idx = pickle.load(f)
    phrase_list = list(phrase2idx.keys())
    embeddings = get_bert_embed(phrase_list, model, tokenizer, summary_method="MEAN", tqdm_bar=True)
    del model
    torch.cuda.empty_cache()
    with open(embedding_path, 'wb') as f:
        np.save(f, embeddings)
    print('start knn')
    similarity, indices = get_KNN(embeddings, 30)
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    with open(similarity_path, 'wb') as f:
        np.save(f, similarity)
    print('done knn')
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_name",
        default="GanjinZero/UMLSBert_ENG",
        type=str,
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="GanjinZero/UMLSBert_ENG",
        type=str,
        help="path to model"
    )
    parser.add_argument(
        "--save_dir",
        default="../use_data/",
        type=str,
        help="output dir"
    )
    parser.add_argument(
        "--phrase2idx_path",
        default="../use_data/phrase2idx.pkl",
        type=str,
        help="Path to phrase2idx file"
    )
    args = parser.parse_args()
    args.indices_path = os.path.join(args.save_dir, 'indices.npy')
    args.similarity_path = os.path.join(args.save_dir, 'similarity.npy')
    args.embedding_path = os.path.join(args.save_dir, 'embedding.npy')
    
    find_new_index(
        tokenizer_name=args.tokenizer_name,
        model_name_or_path=args.model_name_or_path,
        indices_path=args.indices_path,
        similarity_path=args.similarity_path,
        embedding_path=args.embedding_path,
        phrase2idx_path=args.phrase2idx_path
    )
