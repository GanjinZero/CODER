from gensim import models
import os
import sys
sys.path.append("../../")
from pretrain.load_umls import UMLS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from data_util import load
import tqdm

batch_size = 128
device = "cuda:0"


def get_umls():
    umls_label = []
    umls_label_set = set()
    umls_des = []
    umls = UMLS("../../umls", source_range=["MSH", "SNOMEDCT_US", "MDR"], only_load_dict=True)
    for cui in tqdm.tqdm(umls.cui2str):
        if not cui in umls_label_set:
            tmp_str = list(umls.cui2str[cui])
            umls_label.extend([cui] * len(tmp_str))
            umls_des.extend(tmp_str)
            umls_label_set.update([cui])
    print(len(umls_des))
    return umls_label, umls_des


def main(filename, summary_method, umls_label, umls_des):
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

    corpus_list = [("Medline", "es"), ("Medline", "fr"), ("Medline", "nl"), ("Medline", "de"),
                   ("EMEA", "es"), ("EMEA", "fr"), ("EMEA", "nl"), ("EMEA", "de"),
                   ("Patent", "fr"), ("Patent", "de")]
    """
    sty_list = ["Geographic Area",
                "Drug Delivery Device", "Medical Device", "Research Device",
                "Anatomical Abnormality", "Anatomical Structure", "Fully Formed Anatomical Structure",
                "Chemical", "Chemical Viewed Functionally", "Chemical Viewed Structurally", "Inorganic Chemical", "Organic Chemical", "Clinical Drug"]
    """
    result_dict = {}
    umls_embedding = get_bert_embed(umls_des, model, tokenizer, summary_method=summary_method, tqdm_bar=True)

    for corpus in corpus_list:
        output_text, output_label, label_set = load(dataset=corpus[0], lang=corpus[1])
        not_umls_label = [label for label in label_set if not label in umls_label]
        print(f"Count of not appearing in UMLS subset: {len(not_umls_label)}")
        text_embedding = get_bert_embed(output_text, model, tokenizer, summary_method=summary_method)
        predict_label = predict(text_embedding, umls_embedding, umls_label)
        p, r, f1 = metric(output_label, predict_label)
        result_dict[corpus[0] + "|" + corpus[1]] = (p, r, f1)
        print(p, r, f1)

    return result_dict

def predict(text_embedding, umls_embedding, umls_label):
    x_size = text_embedding.size(0)
    sim = torch.matmul(text_embedding, umls_embedding.t())
    most_similar = torch.max(sim, dim=1)[1]
    return [umls_label[idx] for idx in most_similar]


def metric(output_label, predict_label):
    predict_count = 0
    true_count = 0
    correct_count = 0
    for idx in range(len(output_label)):
        if isinstance(predict_label[idx], str):
            predict_label[idx] = [predict_label[idx]]
        if isinstance(output_label[idx], str):
            output_label[idx] = [output_label[idx]]
        predict_count += len(predict_label[idx])
        true_count += len(output_label[idx])
        for pred in predict_label[idx]:
            if pred in output_label[idx]:
                correct_count += 1

    p = correct_count / predict_count
    r = correct_count / true_count
    if p == 0. or r == 0.:
        f1 = 0.
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1


def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm.tqdm(total=count)
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
            if now_count == 0:
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()
    return output


if __name__ == '__main__':
    umls_label, umls_des = get_umls()
    main("bert-base-multilingual-cased", "CLS", umls_label, umls_des)
