import os
import ipdb
from nltk.tokenize import word_tokenize
from icd9 import ICD9
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../../pretrain")
from load_umls import UMLS


tree = ICD9('codes.json')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
log_list = 1 / np.log2(list(range(2, 1001, 1)))

batch_size = 512
max_seq_length = 32


def get_icd9_pairs(icd9_set):
    icd9_pairs = {}
    with open('icd9_grp_file.txt', 'r', encoding="utf-8") as infile:
        data = infile.readlines()
        for row in data:
            codes, name = row.strip().split('#')
            name = name.strip()
            codes = codes.strip().split(' ')
            new_codes = set([])
            for code in codes:
                if code in icd9_set:
                    new_codes.add(code)
                elif len(code) > 5 and code[:5] in icd9_set:
                    new_codes.add(code[:5])
                elif len(code) > 4 and code[:3] in icd9_set:
                    new_codes.add(code[:3])
            codes = list(new_codes)

            if len(codes) > 1:
                for idx, code in enumerate(codes):
                    if code not in icd9_pairs:
                        icd9_pairs[code] = set([])
                    icd9_pairs[code].update(set(codes[:idx]))
                    icd9_pairs[code].update(set(codes[idx+1:]))
    return icd9_pairs


def get_coarse_icd9_pairs(icd9_set):
    icd9_pairs = {}
    ccs_to_icd9 = {}
    with open('ccs_coarsest.txt', 'r', encoding="utf-8") as infile:
        data = infile.readlines()
        currect_ccs = ''
        for row in data:
            if row[:10].strip() != '':
                current_ccs = row[:10].strip()
                ccs_to_icd9[current_ccs] = set([])
            elif row.strip() != '':
                ccs_to_icd9[current_ccs].update(set(row.strip().split(' ')))

    ccs_coarse = {}
    for ccs in list(ccs_to_icd9.keys()):
        ccs_eles = ccs.split('.')
        if len(ccs_eles) >= 2:
            code = ccs_eles[0] + '.' + ccs_eles[1]
            if code not in ccs_coarse:
                ccs_coarse[code] = set([])
            ccs_coarse[code].update(ccs_to_icd9[ccs])

    for ccs in list(ccs_coarse.keys()):
        new_codes = set([])
        for code in ccs_coarse[ccs]:
            if len(code) > 3:
                new_code = code[:3] + '.' + code[3:]
            code = new_code
            if code in icd9_set:
                new_codes.add(code)
            elif len(code) > 5 and code[:5] in icd9_set:
                new_codes.add(code[:5])
            elif len(code) > 4 and code[:3] in icd9_set:
                new_codes.add(code[:3])
        codes = list(new_codes)
        if len(codes) > 1:
            for idx, code in enumerate(codes):
                if code not in icd9_pairs:
                    icd9_pairs[code] = set([])
                icd9_pairs[code].update(set(codes[:idx]))
                icd9_pairs[code].update(set(codes[idx+1:]))
    return icd9_pairs


def get_cui_concept_mappings():
    concept_to_cui_hdr = '2b_concept_ID_to_CUI.txt'
    concept_to_cui = {}
    cui_to_concept = {}
    with open(concept_to_cui_hdr, 'r', encoding="utf-8") as infile:
        lines = infile.readlines()
        for line in lines:
            concept = line.split('\t')[0]
            cui = line.split('\t')[1].split('\r')[0].strip()
            concept_to_cui[concept] = cui
            cui_to_concept[cui] = concept
    return concept_to_cui, cui_to_concept


def get_icd9_reverse_dict(icd9_dict):
    reverse_dict = {}
    for key, value in icd9_dict.items():
        for v in value:
            reverse_dict[v] = key
    return reverse_dict


def get_icd9_cui_mappings():
    cui_to_icd9 = {}
    icd9_to_cui = {}
    with open('cui_icd9.txt', 'r', encoding="utf-8") as infile:
        data = infile.readlines()
        for row in data:
            ele = row.strip().split('|')
            if ele[11] == 'ICD9CM':
                cui = ele[0]
                icd9 = ele[10]
                if cui not in cui_to_icd9 and icd9 != '' and '-' not in icd9:
                    cui_to_icd9[cui] = icd9
                    icd9_to_cui[icd9] = cui
    return cui_to_icd9, icd9_to_cui


def get_icd9_to_description():
    icd9_to_description = {}
    with open('CMS32_DESC_LONG_DX.txt', 'r', encoding='latin-1') as infile:
        data = infile.readlines()
        for row in data:
            icd9 = row.strip()[:6].strip()
            if len(icd9) > 3:
                icd9 = icd9[:3] + '.' + icd9[3:]
            description = row.strip()[6:].strip()
            icd9_to_description[icd9] = description
    return icd9_to_description


def mrm_ccs(embedding_list, embedding_type_list, k=40, check_intersection=False):
    cui_to_icd9, icd9_to_cui = get_icd9_cui_mappings()

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

    umls = UMLS("../../umls", only_load_dict=True)

    if check_intersection:
        cui_list = [cui for cui in intersection_cui
                    if cui in list(cui_to_icd9.keys())]
    else:
        cui_list = list(cui_to_icd9.keys())

    icd9_list = [cui_to_icd9[cui] for cui in cui_list]
    icd9_set = set(icd9_list)
    icd9_pair = get_icd9_pairs(icd9_set)
    icd9_coarse_pair = get_coarse_icd9_pairs(icd9_set)
    icd9_to_description = get_icd9_to_description()

    #icd9_reverse_dict_pair = get_icd9_reverse_dict(icd9_pair)
    #icd9_reverse_dict_coarse_pair = get_icd9_reverse_dict(icd9_coarse_pair)

    #ipdb.set_trace()

    # type label
    # Only part of the icd is calculated as center
    # icd9_to_check = set(icd9_pairs.keys())
    # icd9_to_check.intersection_update(set(icd9_to_idx.keys()))
    pair_center_label = []
    #pair_label = []
    coarse_pair_center_label = []
    #coarse_pair_label = []
    for cui in cui_list:
        if cui_to_icd9[cui] in icd9_pair:
            pair_center_label.append(1)
        else:
            pair_center_label.append(0)
        #pair_label.append(icd9_reverse_dict_pair[cui_to_icd9[cui]])

        if cui_to_icd9[cui] in icd9_coarse_pair:
            coarse_pair_center_label.append(1)
        else:
            coarse_pair_center_label.append(0)
        #coarse_pair_label.append(icd9_reverse_dict_coarse_pair[cui_to_icd9[cui]])

    # generate_description
    description = []
    for cui in cui_list:
        if cui in cui_to_icd9 and cui_to_icd9[cui] in icd9_to_description:
            description.append(icd9_to_description[cui_to_icd9[cui]])
        elif cui in cui_to_icd9 and tree.find(cui_to_icd9[cui]):
            description.append(tree.find(cui_to_icd9[cui]).description)
        elif cui in umls.cui2str:
            description.append(list(umls.cui2str[cui])[0])
        else:
            description.append("")
            print(f"Can not find description for {cui}")

    #ipdb.set_trace()

    opt = []
    for index, embedding in enumerate(embedding_list):
        print("*************************")
        if embedding_type_list[index].lower() == "cui":
            opt.append(mrm_ccs_cui(embedding, icd9_list, cui_list, pair_center_label, icd9_pair, k))
            opt.append(mrm_ccs_cui(embedding, icd9_list, cui_list, coarse_pair_center_label, icd9_coarse_pair, k))
        if embedding_type_list[index].lower() == "word":
            opt.append(mrm_ccs_word(embedding, icd9_list, description, pair_center_label, icd9_pair, k))
            opt.append(mrm_ccs_word(embedding, icd9_list, description, coarse_pair_center_label, icd9_coarse_pair, k))
        if embedding_type_list[index].lower() == "bert":
            opt.append(mrm_ccs_bert(embedding, icd9_list, description, pair_center_label, icd9_pair, k, summary_method="MEAN"))
            opt.append(mrm_ccs_bert(embedding, icd9_list, description, coarse_pair_center_label, icd9_coarse_pair, k, summary_method="MEAN"))
            opt.append(mrm_ccs_bert(embedding, icd9_list, description, pair_center_label, icd9_pair, k, summary_method="CLS"))
            opt.append(mrm_ccs_bert(embedding, icd9_list, description, coarse_pair_center_label, icd9_coarse_pair, k, summary_method="CLS"))
    return opt


def mrm_ccs_cui(cui_embedding, icd9_list, cui_list, center_label, pair, k=40):
    w, _ = load_embedding(cui_embedding)
    print(f"All cui count:{len(cui_list)}")
    new_cui_list = []
    #new_label = []
    new_center_label = []
    new_icd9_list = []
    for index, cui in enumerate(cui_list):
        if cui in w:
            new_cui_list.append(cui)
            new_center_label.append(center_label[index])
            new_icd9_list.append(icd9_list[index])
            #new_label.append(label[index])
    #print(f"Check cui count:{len(new_cui_list)}")

    term_embedding = np.array([w[cui] for cui in new_cui_list])

    return calculate_mrm_ccs(term_embedding, new_icd9_list, new_center_label, pair, k=k)


def mrm_ccs_word(word_embedding, icd9_list, description, center_label, pair, k=40):
    w, dim = load_embedding(word_embedding)

    print(f"All cui count:{len(description)}")
    #cui_str = [[word for word in word_tokenize(
    #    list(umls.cui2str[cui])[0]) if word in w] for cui in cui_list]
    cui_str = []
    #new_label = []
    new_center_label = []
    new_icd9_list = []
    for index, des in enumerate(description):
        tokenize_result = [word for word in word_tokenize(des) if word in w]
        if len(tokenize_result) > 0:
            cui_str.append(tokenize_result)
            new_center_label.append(center_label[index])
            #new_label.append(label[index])
            new_icd9_list.append(icd9_list[index])

    check_count = 0
    for index, cui in tqdm(enumerate(cui_str)):
            tmp_emb = np.zeros((dim))
            for word in cui:
                tmp_emb += w[word]

            if check_count == 0:
                term_embedding = tmp_emb
            else:
                term_embedding = np.concatenate(
                    (term_embedding, tmp_emb), axis=0)
            check_count += 1
    term_embedding = term_embedding.reshape((-1, dim))

    #print(f"Check cui count:{check_count}")

    return calculate_mrm_ccs(term_embedding, new_icd9_list, new_center_label, pair, k=k)


def mrm_ccs_bert(bert_embedding, icd9_list, description, center_label, pair, k=40, summary_method="MEAN"):
    #print(f"Check cui count:{len(description)}")
    model, tokenizer = load_bert(bert_embedding)
    model.eval()

    input_ids = []
    for des in tqdm(description):
        input_ids.append(tokenizer.encode_plus(
            des, max_length=max_seq_length, add_special_tokens=True,
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
                    term_embedding = np.concatenate(
                        (term_embedding, embed_np), axis=0)
                update = min(now_count + batch_size, count) - now_count
                now_count = now_count + update
                pbar.update(update)

    return calculate_mrm_ccs(term_embedding, icd9_list, center_label, pair, k=k)


def calculate_mrm_ccs(term_embedding, icd9_list, center_label, pair, k, normalize=True):
    # term_embedding: term_count * embedding_dim
    # term_type: term_count
    term_embedding = torch.FloatTensor(term_embedding).to(device)
    embedding_norm = torch.norm(
        term_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    term_embedding = torch.div(term_embedding, embedding_norm)
    del embedding_norm

    output = []
    check_count = 0

    count = {}
    for icd9 in tqdm(pair):
        count[icd9] = 0
        for v in pair[icd9]:
            if v in icd9_list:
                count[icd9] += 1

    for index, icd9 in tqdm(enumerate(icd9_list)):
        if center_label[index] == 1 and icd9 in pair:
            now = term_embedding[index]
            score = 0.0
            similarity = torch.matmul(term_embedding, now)
            # The most similar term is itself
            _, indices = torch.topk(similarity, k=k + 1)
            group = pair[icd9]
            for i in range(1, k + 1, 1):
                if icd9_list[indices[i]] in group:
                    score += log_list[i - 1]
            if normalize:
                if score > 0:
                    score /= sum(log_list[0:min(k, count[icd9])])
            output.append(score)
            check_count += 1
    del term_embedding

    if len(output) >= 1:
        score = sum(output) / len(output)
    else:
        score = 0.
    print(f"Check count: {check_count}")
    print(score)
    return score


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
    
    embedding_list = ["../../embeddings/claims_codes_hs_300.txt",
                      "../../embeddings/GoogleNews-vectors-negative300.bin",
                      "../../models/2020_eng"]
    embedding_type_list = ["cui", "word", "bert"]
    mrm_ccs(embedding_list, embedding_type_list)#, normalize=True)
    """
    embedding_list = ["../../embeddings/wikipedia-pubmed-and-PMC-w2v.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin",
                      "/home/yz/pretraining_models/cui2vec.pkl",
                      "../../embeddings/DeVine_etal_200.txt"]
    embedding_type_list = ["word", "word", "word", "cui", "cui"]
    mrm_ccs(embedding_list[3:], embedding_type_list[3:])
    
    embedding_list = ["../../models/2020_all",
                      "/home/yz/pretraining_models/bert-base-cased",
                      "/home/yz/pretraining_models/biobert_v1.1",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                      "/home/yz/pretraining_models/kexinghuang_clinical",
                      "emilyalsentzer/Bio_ClinicalBERT"]
    """
    #mrm_ccs(embedding_list, ["bert"] * 7)
    #mrm_ccs([embedding_list[6]], ["bert"])