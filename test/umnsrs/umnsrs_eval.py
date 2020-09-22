import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from scipy.stats.stats import pearsonr, spearmanr


batch_size = 32
device = "cuda:0"


def main():
    filename = sys.argv[1]
    print(filename)

    if filename[-3:] in ["vec", "txt", "pkl", "bin"]:
        model = load_vectors(filename)
        tokenizer = None
    else:

        try:
            config = AutoConfig.from_pretrained(filename)
            model = AutoModel.from_pretrained(filename, config=config).to(device)
        except BaseException:
            model = torch.load(os.path.join(filename, 'pytorch_model.bin')).to(device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(filename)
        except BaseException:
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(filename, "../"))

    eval(model, tokenizer, './umnsrs/data/umnsrs-rel.txt')
    eval(model, tokenizer, './umnsrs/data/umnsrs-sim.txt')
    eval(model, tokenizer, './umnsrs/data/umnsrs-sim_noRepeat.txt')

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
            vec = np.array(map(float, toks[1:]))
            W[w] = vec
    return W


"""
def get_bert_embed(phrase, m, tok):
    input_id = tok.encode_plus(
        phrase, max_length=32, add_special_tokens=True,
        truncation=True, pad_to_max_length=True, return_tensors="pt")['input_ids']
    m.eval()
    with torch.no_grad():
        embed = m(input_id)[1].cpu().detach()[0].numpy()
    return embed / (norm(embed) + 1e-9)
"""

def norm(v):
    return np.dot(v, v)**0.5

def sim(v0, v1):
    return np.dot(v0, v1)

def embed(phrase, dim, W):
    words = phrase.split()
    vectors = [W[w] for w in words if (w in W)]
    v = sum(vectors, np.zeros(dim))
    return v / (norm(v) + 1e-9)

def get_simlarity_bert(input_0, input_1, m, tok):
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
            embed_0_norm = torch.norm(embed_0, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_0 = torch.div(embed_0, embed_0_norm)
            embed_1_norm = torch.norm(embed_1, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_1 = torch.div(embed_1, embed_1_norm)

            embed_0_mean = torch.mean(m(input_gpu_0)[0], dim=1)
            embed_1_mean = torch.mean(m(input_gpu_1)[0], dim=1)
            embed_0_mean_norm = torch.norm(embed_0_mean, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_0_mean = torch.div(embed_0_mean, embed_0_mean_norm)
            embed_1_mean_norm = torch.norm(embed_1_mean, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embed_1_mean = torch.div(embed_1_mean, embed_1_mean_norm)

            if now_count == 0:
                sim = torch.sum(torch.mul(embed_0, embed_1), dim=1)
                sim_mean = torch.sum(torch.mul(embed_0_mean, embed_1_mean), dim=1)
                #print(sim)
            else:
                tmp = torch.sum(torch.mul(embed_0, embed_1), dim=1)
                tmp_mean = torch.sum(torch.mul(embed_0_mean, embed_1_mean), dim=1)
                sim = torch.cat((sim, tmp), dim=0)
                sim_mean = torch.cat((sim_mean, tmp_mean), dim=0)
                #print(tmp)

            now_count = min(now_count + batch_size, count)
    return sim.cpu().detach().numpy(), sim_mean.cpu().detach().numpy()

def get_simlarity(input_0, input_1, m, dim):
    emb_0 = np.array([embed(x, dim, m) for x in input_0])
    emb_1 = np.array([embed(x, dim, m) for x in input_1])
    return np.sum(emb_0 * emb_1, axis=1)


def eval(m, tok, task_name):
    with open(task_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    ys = [float(line[2]) for line in lines]

    input_term = []
    for line in lines:
        input_term.append(line[0])
    for line in lines:
        input_term.append(line[1])

    if tok is not None:
        preds_cls, preds_mean = get_simlarity_bert(
            input_term[0:len(lines)], input_term[len(lines):], m, tok)
        c_cls, p_cls = spearmanr(preds_cls, ys)
        print(task_name, "CLS", c_cls, p_cls)
        c_mean, p_mean = spearmanr(preds_mean, ys)
        print(task_name, "MEAN", c_mean, p_mean)
    else:
        try:
            dim = m.values()[0].shape[0]
        except BaseException:
            try:
                dim = m.vector_size
            except BaseException:
                dim = 300
        preds = get_simlarity(
            input_term[0:len(lines)], input_term[len(lines):], m, dim)
        c, p = spearmanr(preds, ys)
        print(task_name, c, p)



if __name__ == "__main__":
    main()
