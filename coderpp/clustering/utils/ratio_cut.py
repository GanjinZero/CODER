import pickle
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import spectral_clustering, KMeans
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import argparse



device = torch.device("cuda:0")
batch_size = 64
MAX_CLUSTER_COUNT = 50
mode = 'ratio'
model = AutoModel.from_pretrained('GanjinZero/coder_eng_pp')
tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    return df

def get_bert_embed(phrase_list, normalize=True, summary_method="MEAN", tqdm_bar=False):
    global model
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding='max_length')['input_ids'])
        # print(len(input_ids))
    model.eval()
    model = model.to(device)

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
                embed = model(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0)[0], dim=1)
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
    return np.mean(np.concatenate(output_list, axis=0), axis=0)


def re_cluster(terms_list, mode, similarity, threshold):
    ready = [terms_list]
    res = []
    while ready:
        now = ready.pop()
        clu0, clu1 = cut(now, mode, similarity)
        membed_0 = get_bert_embed(clu0)
        membed_1 = get_bert_embed(clu1)
        if np.dot(membed_0, membed_1) > threshold or len(clu0) <= 1 or len(clu1) <= 1:
            res.append(clu0)
            res.append(clu1)
        else:
            # ready.append(clu0)
            # ready.append(clu1)
            if len(clu0) <= MAX_CLUSTER_COUNT:
                res.append(clu0)
            else:
                ready.append(clu0)
            if len(clu1) <= MAX_CLUSTER_COUNT:
                res.append(clu1)
            else:
                ready.append(clu1)
        # for clu in [clu0, clu1]:
        #     if len(clu) <= MAX_CLUSTER_COUNT:
        #         res.append(clu)
        #     else:
        #         ready.append(clu)
    return res

def cut(terms_list, mode, similarity):
    if mode == 'ratio':
        clu0, clu1 = ratio_cut(terms_list, similarity)
    else:
        clu0, clu1 = normalize_cut(terms_list, similarity)
    return clu0, clu1

def get_sim(terms_list, similarity):
    idx = [phrase2id[x] for x in terms_list]
    sim = np.zeros(shape=(len(idx), len(idx)))
    cnt = len(idx)
    for i in range(cnt):
        for j in range(cnt):
            if idx[j] in indices[idx[i]]:
                sim[i][j] = similarity[idx[i]][np.argwhere(indices[idx[i]]==idx[j])]
            elif idx[i] in indices[idx[j]]:
                sim[i][j] = similarity[idx[j]][np.argwhere(indices[idx[j]]==idx[i])]
    return sim

def laplacian(matrix, normalize=False):
    d_val = matrix.sum(axis=0)
    d = np.diag(d_val)
    l = d - matrix
    if normalize:
        d_inverse_root_val = d_val ** (-1/2)
        d_inverse_root = np.diag(d_inverse_root_val)
        l = np.matmul(np.matmul(d_inverse_root, l), d_inverse_root)
    return l

def ratio_cut(terms_list, similarity):
    sim = get_sim(terms_list, similarity)
    l = laplacian(sim)
    u, v = np.linalg.eig(l)
    index = np.argsort(u.real)
    feat = v[:,index[0:2]].real
    feat_norm = np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
    feat = feat / feat_norm
    cluster = KMeans(n_clusters=2).fit_predict(feat)
    clu0 = np.array(terms_list)[cluster==0].tolist()
    clu1 = np.array(terms_list)[cluster==1].tolist()
    return clu0, clu1


def normalize_cut(terms_list, similarity):
    sim = get_sim(terms_list, similarity)
    l = laplacian(sim, True)
    u, v = np.linalg.eig(l)
    index = np.argsort(u.real)
    feat = v[:,index[0:2]].real
    feat_norm = np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
    feat = feat / feat_norm
    cluster = KMeans(n_clusters=2).fit_predict(feat)
    clu0 = np.array(terms_list)[cluster==0].tolist()
    clu1 = np.array(terms_list)[cluster==1].tolist()
    return clu0, clu1

def print_cluster_to_file(f, one_cluster_result):
    for idx, term in enumerate(one_cluster_result):
        f.write(term)
        if idx != len(one_cluster_result) - 1:
            f.write('|')
        else:
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_data_dir",
        default="../use_data/",
        type=str,
        help="Directory to indices and similarity and idx2phrase"
    )
    parser.add_argument(
        "--result_dir",
        default="../result/",
        type=str,
        help="Directory to save clustering result"
    )
    args = parser.parse_args()
    args.indices_path = args.use_data_dir + 'indices.npy'
    args.similarity_path = args.use_data_dir + 'similarity.npy'
    args.idx2phrase_path = args.use_data_dir + 'idx2phrase.pkl'
    args.result_path = args.result_dir + 'clustering_result.pkl'
    args.phrase2idx_path = args.use_data_dir + 'phrase2idx.pkl'


    cluster_res = load_pickle(args.result_path)
    id2phrase = load_pickle(args.idx2phrase_path)
    phrase2id = load_pickle(args.phrase2idx_path)
    similarity = np.load(args.similarity_path)
    indices = np.load(args.indices_path)


    need_cluster_list = []
    need_cluster_length_list = []
    for key in tqdm(cluster_res):
        if len(cluster_res[key]) > MAX_CLUSTER_COUNT:
            need_cluster_list.append(key)
            need_cluster_length_list.append(len(cluster_res[key]))
            # break

    print(len(need_cluster_list))
    print(np.mean(need_cluster_length_list))

    threshold_list = [0.60]
    for threshold in threshold_list:
        print('threshold=', threshold)
        final_res = []
        for key in tqdm(cluster_res):
            if key not in need_cluster_list:
                final_res.append(list(cluster_res[key]))
                # print_cluster_to_file(f, list(cluster_res[key]))
            else:
                re_cluster_list = re_cluster(list(cluster_res[key]), mode, similarity, threshold)
                for cluster in re_cluster_list:
                    final_res.append(cluster)
                    # print_cluster_to_file(f, cluster)
        with open('../result/final_cluster_res.txt', 'w') as f:
            for cluster in tqdm(final_res):
                print_cluster_to_file(f, cluster)
                # f.write('-------')
                # f.write('RATIO\n')
                # f.write(str(re_cluster(list(cluster_res[key]), mode, similarity, threshold))+'\n')

                # f.write('NORMALIZE\n')
                # f.write(str(re_cluster(list(cluster_res[key]), 'normalize', similarity, threshold))+'\n')

