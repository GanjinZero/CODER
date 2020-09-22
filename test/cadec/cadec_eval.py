from gensim import models
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig


batch_size = 64
device = "cuda:1"


def main():
    filename = sys.argv[1]
    print(filename)

    bert_like = False
    if filename[-3:] in ["vec", "txt"]:
        W = load_vectors(filename, dev=False)
    elif filename[-3:] == "bin":
        W = load_vectors_bin(filename)
    else:
        bert_like = True
        try:
            config = AutoConfig.from_pretrained(filename)
            model = AutoModel.from_pretrained(
                filename, config=config).to(device)
        except BaseException:
            model = torch.load(os.path.join(
                filename, 'pytorch_model.bin')).to(device)

        try:
            model.output_hidden_states = False
        except BaseException:
            pass

        try:
            tokenizer = AutoTokenizer.from_pretrained(filename)
        except BaseException:
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(filename, "../"))

    top_k = 3
    if bert_like:
        eval(model, tokenizer, './cadec/data/cadec', top_k=top_k, summary_method="CLS")
        eval(model, tokenizer, './cadec/data/cadec', top_k=top_k, summary_method="MEAN")
        eval(model, tokenizer, './cadec/data/psytar_disjoint_folds', top_k=top_k, summary_method="CLS")
        eval(model, tokenizer, './cadec/data/psytar_disjoint_folds', top_k=top_k, summary_method="MEAN")
    else:
        eval(W, None, './cadec/data/cadec', top_k=top_k)
        eval(W, None, './cadec/data/psytar_disjoint_folds', top_k=top_k)


def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS"):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
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
            embed_np = embed.cpu().detach().numpy()
            if now_count == 0:
                output = embed_np
            else:
                output = np.concatenate((output, embed_np), axis=0)
            now_count = min(now_count + batch_size, count)
    return output


def eval_one(m, tok, folder, top_k, summary_method=None):
    with open(os.path.join(folder, "standard.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    label2id = {line.strip().split(
        "\t")[0]: index for index, line in enumerate(lines)}
    standard_lines = [line.strip().split("\t") for line in lines]
    #standard_feat = np.array([get_bert_embed(text, m, tok) for (label, text) in standard_lines])
    if tok is not None:
        standard_feat = get_bert_embed(
            [text for (label, text) in standard_lines], m, tok, normalize=True, summary_method=summary_method)
    else:
        standard_feat = embed(
            [text for (label, text) in standard_lines], m.vector_size, m)

    with open(os.path.join(folder, "test.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    test_lines = [line.strip().split("\t") for line in lines]
    #test_feat = np.array([get_bert_embed(text, m, tok) for (label, text) in test_lines])
    if tok is not None:
        test_feat = get_bert_embed(
            [text for (label, text) in test_lines], m, tok, normalize=True, summary_method=summary_method)
    else:
        test_feat = embed(
            [text for (label, text) in test_lines], m.vector_size, m)

    sim_mat = np.dot(test_feat, standard_feat.T)

    correct_1 = 0
    correct_k = 0
    pred_top_k = torch.topk(torch.FloatTensor(sim_mat), k=top_k)[
        1].cpu().numpy()
    for i in range(len(test_lines)):
        true_id = label2id[test_lines[i][0]]
        if pred_top_k[i][0] == true_id:
            correct_1 += 1
        if true_id in list(pred_top_k[i]):
            correct_k += 1
    acc_1 = correct_1 / len(test_lines)
    acc_k = correct_k / len(test_lines)
    return acc_1, acc_k


def eval(m, tok, task_name, top_k=3, summary_method=None):
    acc_1_list = []
    acc_k_list = []
    for p in os.listdir(task_name):
        acc_1, acc_k = eval_one(m, tok, os.path.join(task_name, p), top_k, summary_method=summary_method)
        acc_1_list.append(acc_1)
        acc_k_list.append(acc_k)
    print(task_name, summary_method)
    print(f"top_k={top_k}")
    print(acc_1_list)
    print(acc_k_list)
    print(sum(acc_1_list) / 5, sum(acc_k_list) / 5)

    return None


def load_vectors(filename):
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


def load_vectors_bin(filename):
    w = models.KeyedVectors.load_word2vec_format(filename, binary=True)
    return w


def cosine(u, v):
    return np.dot(u, v)


def norm(v):
    return np.dot(v, v)**0.5


def embed_one(phrase, dim, W):
    words = phrase.split()
    vectors = [W[w] for w in words if (w in W)]
    v = sum(vectors, np.zeros(dim))
    return v / (norm(v) + 1e-9)


def embed(phrase_list, dim, W):
    return np.array([embed_one(phrase, dim, W) for phrase in phrase_list])


if __name__ == '__main__':
    main()
