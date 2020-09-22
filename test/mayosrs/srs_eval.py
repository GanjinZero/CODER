import os
import sys
from collections import defaultdict
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from read_data import get_srs, get_srs_cui
from gensim import models
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


batch_size = 32
device = "cuda:0"


def main():
    filename = sys.argv[1]
    embedding_type = sys.argv[2]

    #W = load_vectors(filename, dev=True)
    if filename[-3:] in ["vec", "txt", "pkl", "bin"]:
        model = load_vectors(filename)
        tokenizer = None
    else:
        try:
            config = AutoConfig.from_pretrained(filename)
            model = AutoModel.from_pretrained(
                filename, config=config).to(device)
        except BaseException:
            model = torch.load(os.path.join(
                filename, 'pytorch_model.bin')).to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(filename)
        except BaseException:
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(filename, "../"))
    if filename[-3:] in ["vec", "txt", "pkl", "bin"]:
        try:
            dim = model.values()[0].shape[0]
        except BaseException:
            try:
                dim = model.vector_size
            except BaseException:
                dim = 300
    else:
        if filename.find('large') >= 0:
            dim = 1024
        else:
            dim = 768

    print(filename)

    srs = get_srs()
    srs_cui = get_srs_cui()
    if filename[-3:] in ["vec", "txt", "pkl", "bin"]:
        if embedding_type == "word":
            for task, scores in srs.items():
                xs, ys = scores
                preds = []
                for x in xs:
                    phrase1, phrase2 = x
                    v1 = embed(phrase1, dim, model, embedding_type)
                    v2 = embed(phrase2, dim, model, embedding_type)
                    cos = cosine(v1, v2)
                    preds.append(cos)

                c, p = spearmanr(preds, ys)
                print(task, c, p)
        if embedding_type == "cui":
            for task, scores in srs_cui.items():
                xs, ys = scores
                preds = []
                for x in xs:
                    phrase1, phrase2 = x
                    v1 = embed(phrase1, dim, model, embedding_type)
                    v2 = embed(phrase2, dim, model, embedding_type)
                    cos = cosine(v1, v2)
                    preds.append(cos)

                c, p = spearmanr(preds, ys)
                print(task, c, p)
    else:
        for task, scores in srs.items():
            xs, ys = scores
            preds = []
            input_0 = []
            input_1 = []
            for x in xs:
                phrase1, phrase2 = x
                input_0.append(phrase1)
                input_1.append(phrase2)

            preds_cls, preds_mean = get_simlarity(
                input_0, input_1, model, tokenizer)
            c_cls, p_cls = spearmanr(preds_cls, ys)
            print(task, "CLS", c_cls, p_cls)
            c_mean, p_mean = spearmanr(preds_mean, ys)
            print(task, "MEAN", c_mean, p_mean)


def get_simlarity(input_0, input_1, m, tok):
    input_ids_0 = []
    input_ids_1 = []
    for phrase in input_0:
        input_ids_0.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    for phrase in input_1:
        input_ids_1.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    count = len(input_0)

    now_count = 0
    m.eval()
    with torch.no_grad():
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids_0[now_count:min(
                now_count + batch_size, count)]).to(device)
            input_gpu_1 = torch.LongTensor(input_ids_1[now_count:min(
                now_count + batch_size, count)]).to(device)
            embed_0 = m(input_gpu_0)[1]
            embed_1 = m(input_gpu_1)[1]
            embed_0_norm = torch.norm(
                embed_0, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_0 = torch.div(embed_0, embed_0_norm)
            embed_1_norm = torch.norm(
                embed_1, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_1 = torch.div(embed_1, embed_1_norm)

            embed_0_mean = torch.mean(m(input_gpu_0)[0], dim=1)
            embed_1_mean = torch.mean(m(input_gpu_1)[0], dim=1)
            embed_0_mean_norm = torch.norm(
                embed_0_mean, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_0_mean = torch.div(embed_0_mean, embed_0_mean_norm)
            embed_1_mean_norm = torch.norm(
                embed_1_mean, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_1_mean = torch.div(embed_1_mean, embed_1_mean_norm)

            if now_count == 0:
                sim = torch.sum(torch.mul(embed_0, embed_1), dim=1)
                sim_mean = torch.sum(
                    torch.mul(embed_0_mean, embed_1_mean), dim=1)
                # print(sim)
            else:
                tmp = torch.sum(torch.mul(embed_0, embed_1), dim=1)
                tmp_mean = torch.sum(
                    torch.mul(embed_0_mean, embed_1_mean), dim=1)
                sim = torch.cat((sim, tmp), dim=0)
                sim_mean = torch.cat((sim_mean, tmp_mean), dim=0)
                # print(tmp)

            now_count = min(now_count + batch_size, count)
    return sim.cpu().detach().numpy(), sim_mean.cpu().detach().numpy()


def cosine(u, v):
    return np.dot(u, v)


def norm(v):
    return np.dot(v, v)**0.5


def vec_format(emb):
    if emb[0] == "[":
        return np.array([float(num) for num in emb[1:-1].split(',')])
    return emb


def embed(phrase, dim, W, embedding_type):
    if embedding_type == "word":
        words = phrase.split()
        vectors = [W[w] for w in words if (w in W)]
        v = sum(vectors, np.zeros(dim))
        return v / (norm(v) + 1e-9)
    if embedding_type == "cui":
        if phrase in W:
            #print(W[phrase])
            return vec_format(W[phrase])
        if 'empty' in W:
            return vec_format(W['empty'])
        return np.zeros_like(list(W.values())[0])


def get_bert_embed(phrase, m, tok):
    input_id = tok.encode_plus(
        phrase, max_length=32, add_special_tokens=True,
        truncation=True, pad_to_max_length=True, return_tensors="pt")['input_ids']
    m.eval()
    with torch.no_grad():
        embed = m(input_id)[1].cpu().detach()[0]
    return embed / (norm(embed) + 1e-9)


def load_vectors(filename):
    if filename.find('bin') >= 0:
        from gensim import models
        W = models.KeyedVectors.load_word2vec_format(filename, binary=True)
        return W

    if filename.find('pkl') >= 0:
        import pickle
        with open(filename, 'rb') as f:
            W = pickle.load(f)
        return W

    W = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            toks = line.strip().split()
            w = toks[0]
            vec = np.array(list(map(float, toks[1:])))
            W[w] = vec
    return W


if __name__ == '__main__':
    main()
