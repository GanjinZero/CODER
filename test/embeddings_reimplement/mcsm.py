from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../../pretrain")
from load_umls import UMLS
from nltk.tokenize import word_tokenize
import ipdb
import os


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
log_list = 1 / np.log2(list(range(2, 1001, 1)))

batch_size = 512
max_seq_length = 32

# umls = UMLS("../../umls", source_range='SNOMEDCT_US')
t_list = ['Pharmacologic Substance', 'Disease or Syndrome',
          'Neoplastic Process', 'Clinical Drug', 'Finding', 'Injury or Poisoning']


def mcsm(embedding_list, embedding_type_list, type_list=t_list, k=40, lang_range=['ENG'], check_intersection=False):
    if check_intersection:
        if not os.path.exists("intersection.txt"):
            intersection_cui = get_intersection(
                embedding_list, embedding_type_list)
            with open("intersection.txt", "w", encoding="utf-8") as f:
                for cui in intersection_cui:
                    f.write(cui.strip() + "\n")
        else:
            with open("intersection.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
            intersection_cui = [line.strip() for line in lines]

    umls = UMLS("../../umls", source_range='SNOMEDCT_US',
                lang_range=lang_range)
    if check_intersection:
        cui_list = [cui for cui in intersection_cui
                    if cui in umls.cui2sty and umls.cui2sty[cui] in type_list]
    else:
        cui_list = [cui for cui, sty in umls.cui2sty.items()
                    if sty in type_list]
    opt = []
    for index, embedding in enumerate(embedding_list):
        if embedding_type_list[index].lower() == "cui":
            opt.append(mcsm_cui(embedding, umls, cui_list, type_list, k))
        if embedding_type_list[index].lower() == "word":
            opt.append(mcsm_word(embedding, umls, cui_list, type_list, k))
        if embedding_type_list[index].lower() == "bert":
            opt.append(mcsm_bert(embedding, umls, cui_list,
                                 type_list, k, summary_method="MEAN"))
            opt.append(mcsm_bert(embedding, umls, cui_list,
                                 type_list, k, summary_method="CLS"))
    return opt


def mcsm_cui(cui_embedding, umls, cui_list, type_list, k=40):
    w, _ = load_embedding(cui_embedding)
    if cui_list is None:
        cui_list = list(w.keys())
        print(f"Check cui count:{len(cui_list)}")
    else:
        print(f"All cui count:{len(cui_list)}")
        cui_list = list(set(w.keys()).intersection(set(cui_list)))
        print(f"Check cui count:{len(cui_list)}")

    term_embedding = np.array([w[cui] for cui in cui_list])
    term_type = [umls.cui2sty[cui] for cui in cui_list]

    return calculate_mcsm(term_embedding, term_type, type_list, k=k)


def mcsm_word(word_embedding, umls, cui_list, type_list, k=40):
    w, dim = load_embedding(word_embedding)

    print(f"All cui count:{len(cui_list)}")
    cui_str = [[word for word in word_tokenize(
        list(umls.cui2str[cui])[0]) if word in w] for cui in cui_list]

    check_count = 0
    term_type = []
    for index, cui in tqdm(enumerate(cui_str)):
        if len(cui) > 0:
            term_type.append(umls.cui2sty[cui_list[index]])

            tmp_emb = np.zeros((dim))
            for word in cui:
                tmp_emb += w[word]

            if check_count == 0:
                term_embedding = tmp_emb
            else:
                term_embedding = np.concatenate(
                    (term_embedding, tmp_emb), axis=0)
            check_count += 1
            """
            if check_count > 500:
                break
            """
    term_embedding = term_embedding.reshape((-1, dim))

    print(f"Check cui count:{check_count}")

    return calculate_mcsm(term_embedding, term_type, type_list, k=k)


def mcsm_bert(bert_embedding, umls, cui_list, type_list, k=40, summary_method="MEAN"):
    print(f"Check cui count:{len(cui_list)}")
    model, tokenizer = load_bert(bert_embedding)
    model.eval()

    input_ids = []
    for cui in tqdm(cui_list):
        input_ids.append(tokenizer.encode_plus(
            list(umls.cui2str[cui])[
                0], max_length=max_seq_length, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])

    count = len(input_ids)
    now_count = 0
    with tqdm(total=count) as pbar:
        with torch.no_grad():
            while now_count < count:
                input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                    now_count + batch_size, count)]).to(device)
                if summary_method == "CLS":
                    embed = model(input_gpu_0)[1]
                if summary_method == "MEAN":
                    embed = torch.mean(model(input_gpu_0)[0], dim=1)
                embed_np = embed.cpu().detach().numpy()
                if now_count == 0:
                    term_embedding = embed_np
                else:
                    term_embedding = np.concatenate((term_embedding, embed_np), axis=0)
                update = min(now_count + batch_size, count) - now_count
                now_count = now_count + update
                pbar.update(update)

    term_type = [umls.cui2sty[cui] for cui in cui_list]
    return calculate_mcsm(term_embedding, term_type, type_list, k=k)


def summary(opt):
    new_opt = {k: (np.mean(v), np.std(v)) for k, v in opt.items()}
    return new_opt


def calculate_mcsm(term_embedding, term_type, target_type_list, k):
    # term_embedding: term_count * embedding_dim
    # term_type: term_count
    term_embedding = torch.FloatTensor(term_embedding).to(device)
    embedding_norm = torch.norm(
        term_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    term_embedding = torch.div(term_embedding, embedding_norm)
    del embedding_norm
    output = {target_type: [] for target_type in target_type_list}
    for index, t in tqdm(enumerate(term_type)):
        if t in target_type_list:
            now = term_embedding[index]
            score = 0.0
            similarity = torch.matmul(term_embedding, now)
            # The most similar term is itself
            _, indices = torch.topk(similarity, k=k + 1)
            for i in range(1, k + 1, 1):
                if term_type[indices[i]] == t:
                    score += log_list[i - 1]
            output[t].append(score)
    del term_embedding

    output = summary(output)
    print(output)
    return output


def load_embedding(filename):
    print(filename)
    if filename.find('bin') >= 0:
        from gensim import models
        W = models.KeyedVectors.load_word2vec_format(filename, binary=True)
        dim = W.vector_size
        return W, dim

    if filename.find('pkl') >= 0:
        import pickle
        with open(filename, 'rb') as f:
            W = pickle.load(f)
        for key, value in W.items():
            W[key] = np.array(list(map(float, value[1:-1].split(","))))
        dim = len(list(W.values())[0])
        return W, dim

    W = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            toks = line.strip().split()
            w = toks[0]
            vec = np.array(list(map(float, toks[1:])))
            W[w] = vec
    dim = len(list(W.values())[0])
    return W, dim


def load_bert(model_name_or_path):
    print(model_name_or_path)
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(
            model_name_or_path, config=config).to(device)
    except BaseException:
        model = torch.load(os.path.join(
            model_name_or_path, 'pytorch_model.bin')).to(device)

    try:
        model.output_hidden_states = False
    except BaseException:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except BaseException:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name_or_path, "../"))
    return model, tokenizer


def get_intersection(embedding_list, embedding_type_list):
    intersection_cui = set()
    checker = True
    for index, embed in enumerate(embedding_list):
        if embedding_type_list[index] == "cui":
            w, _ = load_embedding(embed)
            if checker:
                intersection_cui = set(list(w.keys()))
                checker = False
            else:
                intersection_cui = set(
                    list(w.keys())).intersection(intersection_cui)
    print(f"Intersection count: {len(intersection_cui)}")
    return list(intersection_cui)


if __name__ == "__main__":
    """
    embedding_list = ["../../embeddings/claims_codes_hs_300.txt",
                      "../../embeddings/GoogleNews-vectors-negative300.bin",
                      "../../models/2020_eng"]
    #embedding_type_list = ["cui", "word", "bert"]
    embedding_list = ["../../embeddings/wikipedia-pubmed-and-PMC-w2v.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin"]
    embedding_type_list = ["word", "word", "word"]
    embedding_list = ["../../embeddings/DeVine_etal_200.txt",
                      "/home/yz/pretraining_models/cui2vec.pkl"]
    embedding_type_list = ["cui", "cui"]
    """
    #mcsm([embedding_list[2], embedding_type_list[2]])
    """
    embedding_list = ["../../embeddings/claims_codes_hs_300.txt",
                      "../../embeddings/DeVine_etal_200.txt",
                      "/home/yz/pretraining_models/cui2vec.pkl"]
    embedding_type_list = ["cui", "cui", "cui"]
    mcsm(embedding_list, embedding_type_list, check_intersection=True)
    """
    #embedding_list = ["../../models/2020_eng", "../../models/2020_all"]
    #mcsm(embedding_list, ["bert"] * 2, check_intersection=True)

    """
    embedding_list = ["../../embeddings/wikipedia-pubmed-and-PMC-w2v.bin",
                      "../../embeddings/GoogleNews-vectors-negative300.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin"]
    mcsm(embedding_list, ["word"] * 4, check_intersection=True)
    """

    embedding_list = ["/home/yz/pretraining_models/bert-base-cased",
                      "/home/yz/pretraining_models/biobert_v1.1",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                      "/home/yz/pretraining_models/kexinghuang_clinical",
                      "emilyalsentzer/Bio_ClinicalBERT",
                      "../../models/UMLSBert_nosty"]
    #mcsm(embedding_list, ["bert"] * 6, check_intersection=True)
    #mcsm(embedding_list, ["bert"] * 6)
    mcsm([embedding_list[-1]], ["bert"])
