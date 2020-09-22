from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../../pretrain")
from load_umls import UMLS
from nltk.tokenize import word_tokenize
import os
import ipdb


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 512
max_seq_length = 32

def get_drug_diseases_to_check(concept_filename):
    query_to_targets = {}
    with open(concept_filename, 'r') as infile:
        data = infile.readlines()
        for row in data:
            drug, diseases = row.strip().split(':')
            diseases = diseases.split(',')[:-1]
            disease_cui_set = set([])
            for disease in diseases:
                disease_cui_set.add(disease)
            if len(disease_cui_set) > 0:
                query_to_targets[drug] = disease_cui_set

    cui_list = set()
    for query, targets in query_to_targets.items():
        cui_list.update([query])
        cui_list.update(targets)
    return query_to_targets, list(cui_list)

def normalize(tensor):
    norm = torch.norm(tensor, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return torch.div(tensor, norm)

def calculate_mrm_ndfrt_origin(term_embedding, cui_list, query_to_targets, k):
    return calculate_mrm_ndfrt_delta(term_embedding, cui_list, query_to_targets, None, k)


def calculate_mrm_ndfrt_q2t(term_embedding, cui_list, query_to_targets, k):
    delta_list = []

    term_embedding = torch.FloatTensor(term_embedding).to(device)
    norm_embedding = normalize(term_embedding)

    id2cui = {i:cui_list[i] for i in range(len(cui_list))}
    cui2id = {cui:index for index, cui in id2cui.items()}

    for query, targets in query_to_targets.items():
        if query in cui2id:
            for target in targets:
                if target in cui2id:
                    delta = term_embedding[cui2id[query]] - term_embedding[cui2id[target]]
                    delta_list.append(delta)

    overall_output = []
    for _, delta in tqdm(enumerate(delta_list)):
        output = []
        for query, targets in query_to_targets.items():
            if query in cui2id:
                find_embedding = term_embedding[cui2id[query]] - delta
                similarity = torch.matmul(norm_embedding, find_embedding)
                _, indices = torch.topk(similarity, k=k + 1)
                find_cui = [cui_list[index] for index in indices[1:]]
                score = 0.
                for cui in find_cui:
                    if cui in targets:
                        score = 1.
                        break
                output.append(score)
        if len(output) > 0:
            score = sum(output) / len(output)
        else:
            score = 0.  
        overall_output.append(score)

    if len(overall_output) > 0:
        overall_score = sum(overall_output) / len(overall_output)
        overall_max = max(overall_output)
    else:
        overall_score = 0
        overall_max = 0
    return overall_score, overall_max


def calculate_mrm_ndfrt_delta(term_embedding, cui_list, query_to_targets, delta=None, k=40):
    term_embedding = torch.FloatTensor(term_embedding).to(device)
    norm_embedding = normalize(term_embedding)

    id2cui = {i:cui_list[i] for i in range(len(cui_list))}
    cui2id = {cui:index for index, cui in id2cui.items()}

    output = []
    check_count = 0
    for query, targets in query_to_targets.items():
        if query in cui2id:
            query_embedding = term_embedding[cui2id[query]]
            if delta is None:
                find_embedding = query_embedding
            else:
                find_embedding = query_embedding - torch.FloatTensor(delta).to(device)
            similarity = torch.matmul(norm_embedding, find_embedding)
            _, indices = torch.topk(similarity, k=k + 1)
            find_cui = [cui_list[index] for index in indices[1:]]
            score = 0.
            for cui in find_cui:
                if cui in targets:
                    score = 1.
                    break
            output.append(score)
            check_count += 1
    del term_embedding

    if len(output) > 0:
        score = sum(output) / len(output)
    else:
        score = 0.

    """
    print(f"Check count: {check_count}")
    print(score)
    """

    return score


def mrm_ndfrt_cui(cui_embedding, umls, cui_list, query_to_targets, k, method):
    w, _ = load_embedding(cui_embedding)

    new_cui_list = [cui for cui in cui_list if cui in w]
    term_embedding = np.array([w[cui] for cui in new_cui_list])

    print(f"Cui count:{len(new_cui_list)}")

    if method == "origin":
        score = calculate_mrm_ndfrt_origin(term_embedding, new_cui_list, query_to_targets, k)
        print(f"Origin: {score}")
    if method == "all":
        score = calculate_mrm_ndfrt_q2t(term_embedding, new_cui_list, query_to_targets, k)
        average_score, max_score = score
        print(f"Average: {average_score}")
        print(f"Max: {max_score}")
    return score


def mrm_ndfrt_word(word_embedding, umls, cui_list, query_to_targets, k, method):
    w, dim = load_embedding(word_embedding)

    print("Tokenize and calculate avg embedding.")
    cui_str = [[word for word in word_tokenize(
        list(umls.cui2str[cui])[0]) if word in w] for cui in cui_list if cui in umls.cui2str]

    new_cui_list = []
    check_count = 0
    for index, des in enumerate(cui_str):
        if len(des) > 0:
            tmp_emb = np.zeros((dim))
            for word in des:
                tmp_emb += w[word]

            if check_count == 0:
                term_embedding = tmp_emb
            else:
                term_embedding = np.concatenate(
                    (term_embedding, tmp_emb), axis=0)
            check_count += 1
            new_cui_list.append(cui_list[index])
    term_embedding = term_embedding.reshape((-1, dim))

    print(f"Cui count:{len(new_cui_list)}")

    if method == "origin":
        score = calculate_mrm_ndfrt_origin(term_embedding, new_cui_list, query_to_targets, k)
        print(f"Origin: {score}")
    if method == "all":
        score = calculate_mrm_ndfrt_q2t(term_embedding, new_cui_list, query_to_targets, k)
        average_score, max_score = score
        print(f"Average: {average_score}")
        print(f"Max: {max_score}")
    return score


def mrm_ndfrt_bert(bert_embedding, umls, cui_list, query_to_targets, k, method, summary_method):
    print(summary_method)
    model, tokenizer = load_bert(bert_embedding)
    model.eval()

    input_ids = []
    new_cui_list = []
    for cui in cui_list:
        if cui in umls.cui2str:
            input_ids.append(tokenizer.encode_plus(
                list(umls.cui2str[cui])[
                    0], max_length=max_seq_length, add_special_tokens=True,
                truncation=True, pad_to_max_length=True)['input_ids'])
            new_cui_list.append(cui)

    count = len(input_ids)
    now_count = 0
    # with tqdm(total=count) as pbar:
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
            # pbar.update(update)

    print(f"Cui count:{len(new_cui_list)}")

    if method == "origin":
        score = calculate_mrm_ndfrt_origin(term_embedding, new_cui_list, query_to_targets, k)
        print(f"Origin: {score}")
    if method == "all":
        score = calculate_mrm_ndfrt_q2t(term_embedding, new_cui_list, query_to_targets, k)
        average_score, max_score = score
        print(f"Average: {average_score}")
        print(f"Max: {max_score}")
    if method in ["may_treat", "may_prevent"]:
        beta_path = os.path.join(bert_embedding, "run", "1000000", "rel embedding")
        with open(os.path.join(beta_path, "metadata.tsv"), "r", encoding="utf-8") as f:
            metadata = f.readlines()
        metadata = [line.strip() for line in metadata]
        with open(os.path.join(beta_path, "tensors.tsv"), "r", encoding="utf-8") as f:
            tensor = f.readlines()

        tensor = [[float(num) for num in line.split("\t")] for line in tensor]
        for index, title in enumerate(metadata):
            if title == method:
                delta = tensor[index]
        
        score = calculate_mrm_ndfrt_delta(term_embedding, new_cui_list, query_to_targets, delta, k)
        print(f"{method}: {score}")
    return score


def mrm_ndfrt(embedding_list, embedding_type_list, concept_filename, k=40, check_intersection=True):
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

    query_to_targets, cui_list = get_drug_diseases_to_check(concept_filename)
    umls = UMLS("../../umls", only_load_dict=True) # source_range='SNOMEDCT_US')#, only_load_dict=True)

    if check_intersection:
        cui_list = [cui for cui in cui_list if cui in intersection_cui]

    #cui_list = [cui for cui in umls.cui2str if umls.cui2sty[cui] in sty_list]
    #cui_list = [cui for cui in cui_list if cui in umls.sty_list]

    """
    for cui in cui_list:
        if not cui in umls.cui2str:
            print(cui)

    ipdb.set_trace()
    """

    opt = []
    """
    # Origin
    print("ORIGIN")
    for index, embedding in enumerate(embedding_list):
        if embedding_type_list[index].lower() == "cui":
            opt.append(mrm_ndfrt_cui(embedding, umls, cui_list, query_to_targets, k, "origin"))
        if embedding_type_list[index].lower() == "word":
            opt.append(mrm_ndfrt_word(embedding, umls, cui_list, query_to_targets, k, "origin"))
        if embedding_type_list[index].lower() == "bert":
            #opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
            #                     query_to_targets, k, "origin", summary_method="MEAN"))
            opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                 query_to_targets, k, "origin", summary_method="CLS"))

    # For UMLSBert
    for index, embedding in enumerate(embedding_list):
        if embedding_type_list[index].lower() == "bert":
            print("BETA")
            beta_path = os.path.join(embedding, "run", "1000000", "rel embedding")
            if os.path.exists(beta_path):
                if concept_filename.find('treat') >= 0:
                    method = "may_treat"
                else:
                    method = "may_prevent"
                #opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                #                 query_to_targets, k, method, summary_method="MEAN"))
                opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                 query_to_targets, k, method, summary_method="CLS"))                

    # For average and max

    print("ALL")
    for index, embedding in enumerate(embedding_list):
        if embedding_type_list[index].lower() == "cui":
            opt.append(mrm_ndfrt_cui(embedding, umls, cui_list, query_to_targets, k, "all"))
        if embedding_type_list[index].lower() == "word":
            opt.append(mrm_ndfrt_word(embedding, umls, cui_list, query_to_targets, k, "all"))
        if embedding_type_list[index].lower() == "bert":
            #opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
            #                     query_to_targets, k, "all", summary_method="MEAN"))
            opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                 query_to_targets, k, "all", summary_method="CLS"))
    """
    for index, embedding in enumerate(embedding_list):
        if embedding_type_list[index].lower() == "cui":
            opt.append(mrm_ndfrt_cui(embedding, umls, cui_list, query_to_targets, k, "origin"))
            opt.append(mrm_ndfrt_cui(embedding, umls, cui_list, query_to_targets, k, "all"))
        if embedding_type_list[index].lower() == "word":
            opt.append(mrm_ndfrt_word(embedding, umls, cui_list, query_to_targets, k, "origin"))
            opt.append(mrm_ndfrt_word(embedding, umls, cui_list, query_to_targets, k, "all"))
        if embedding_type_list[index].lower() == "bert":
            opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                      query_to_targets, k, "origin", summary_method="CLS"))
            beta_path = os.path.join(embedding, "run", "1000000", "rel embedding")
            if os.path.exists(beta_path):
                if concept_filename.find('treat') >= 0:
                    method = "may_treat"
                else:
                    method = "may_prevent"  
                opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                            query_to_targets, k, method, summary_method="CLS"))
            opt.append(mrm_ndfrt_bert(embedding, umls, cui_list,
                                      query_to_targets, k, "all", summary_method="CLS"))                   

    return opt

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
    embedding_type_list = ["cui", "word", "bert"]
    mrm_ndfrt(embedding_list, embedding_type_list, "may_prevent_cui.txt", check_intersection=False)
    """

    embedding_list = ["../../embeddings/wikipedia-pubmed-and-PMC-w2v.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin",
                      "../../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin"]
    embedding_type_list = ["word", "word", "word"]
    embedding_list += ["../../embeddings/DeVine_etal_200.txt",
                      "/home/yz/pretraining_models/cui2vec.pkl"]
    embedding_type_list += ["cui", "cui"]
    embedding_list += ["../../models/2020_all",
                      "/home/yz/pretraining_models/bert-base-cased",
                      "/home/yz/pretraining_models/biobert_v1.1",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract",
                      "/home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                      "/home/yz/pretraining_models/kexinghuang_clinical",
                      "emilyalsentzer/Bio_ClinicalBERT"]
    embedding_type_list += ["bert"] * 7

    #mrm_ndfrt(embedding_list, embedding_type_list, "may_treat_cui.txt", check_intersection=True)
    mrm_ndfrt(embedding_list, embedding_type_list, "may_treat_cui.txt", check_intersection=False)
    #mrm_ndfrt(embedding_list[-6:], embedding_type_list[-6:], "may_prevent_cui.txt", check_intersection=True)
    #mrm_ndfrt(embedding_list, embedding_type_list, "may_prevent_cui.txt", check_intersection=False)