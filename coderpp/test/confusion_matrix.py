import pickle
import numpy as np
import itertools
from tqdm import tqdm
from load_umls import UMLS
import ahocorasick
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os
import argparse
import random

def eval_clustering(umls, similarity, indices, threshold, args):
    with open(args.idx2phrase_path, 'rb') as f:
        idx2phrase = pickle.load(f)
    gt_clustering = list(umls.cui2str.values())
    confusion_matrix = np.zeros((2, 2))
    # in a group
    TP = ahocorasick.Automaton()
    similarity_list_actual_p = []
    similarity_list_actual_n = []
    T = ahocorasick.Automaton()
    # with open(args.output_dir+'FN_finetune.txt', 'w') as f:
    if True:
        for group in tqdm(gt_clustering):
            if len(group) < 2:
                continue
            group = list(group)
            for pair in itertools.combinations(group, r=2):
                if str(pair[0])+str(pair[1])+idx2phrase[pair[0]]+idx2phrase[pair[1]] in T or str(pair[1])+str(pair[0])+idx2phrase[pair[1]]+idx2phrase[pair[0]] in T:
                    continue
                T.add_word(str(pair[0])+str(pair[1])+idx2phrase[pair[0]]+idx2phrase[pair[1]], '')
                T.add_word(str(pair[1])+str(pair[0])+idx2phrase[pair[1]]+idx2phrase[pair[0]], '')
                if pair[1] in indices[pair[0]]:
                    index = np.where(indices[pair[0]] == pair[1])
                    similarity_list_actual_p.append(min(similarity[pair[0], index][0][0], 1))
                    if similarity[pair[0], index] > threshold:
                        confusion_matrix[0, 0] += 1
                        TP.add_word(str(pair[0])+str(pair[1])+idx2phrase[pair[0]]+idx2phrase[pair[1]], '')
                        TP.add_word(str(pair[1])+str(pair[0])+idx2phrase[pair[1]]+idx2phrase[pair[0]], '')
                    else:
                        confusion_matrix[0, 1] += 1
                        # f.write(idx2phrase[pair[0]]+', '+idx2phrase[pair[1]]+', '+'\t'+str(similarity[pair[0], index][0][0])+'\n')
                elif pair[0] in indices[pair[1]]:
                    index = np.where(indices[pair[1]] == pair[0])
                    similarity_list_actual_p.append(min(similarity[pair[1], index][0][0], 1))
                    if similarity[pair[1], index] > threshold:
                        confusion_matrix[0, 0] += 1
                        TP.add_word(str(pair[0])+str(pair[1])+idx2phrase[pair[0]]+idx2phrase[pair[1]], '')
                        TP.add_word(str(pair[1])+str(pair[0])+idx2phrase[pair[1]]+idx2phrase[pair[0]], '')
                    else:
                        confusion_matrix[0, 1] += 1
                        # f.write(idx2phrase[pair[0]]+', '+idx2phrase[pair[1]]+', '+'\t'+str(similarity[pair[1], index][0][0])+'\n')
                else:
                    confusion_matrix[0, 1] += 1
                    # f.write(idx2phrase[pair[0]]+', '+idx2phrase[pair[1]]+'\n')


    # not in a group
    predicted_p = 0
    fp = 0
    fp_list = []
    A = ahocorasick.Automaton()
    for string in tqdm(umls.stridx_list):
        A.add_word(str(string), str(string))
    # with open(args.output_dir+'FP_finetune.txt', 'w') as f:   
    if True:
        for i in tqdm(umls.stridx_list):
            for j in range(1, indices.shape[1]):
                if idx2phrase[i] != idx2phrase[indices[i][j]] and str(indices[i][j]) in A and str(i)+str(indices[i][j])+idx2phrase[i]+idx2phrase[indices[i][j]] not in T and similarity[i][j] > 0:
                    similarity_list_actual_n.append(min(similarity[i][j], 1))
                    if i in indices[indices[i][j]]:
                        index = np.where(indices[indices[i][j]] == i)
                        similarity[indices[i][j]][index] = 0

                if similarity[i][j] > threshold and idx2phrase[i] != idx2phrase[indices[i][j]] and str(indices[i][j]) in A:
                    predicted_p += 1
                    if str(i)+str(indices[i][j])+idx2phrase[i]+idx2phrase[indices[i][j]] not in TP:
                        fp += 1
                        # print((idx2phrase[i], idx2phrase[indices[i][j]]))
                        # f.write(idx2phrase[i]+'\t'+idx2phrase[indices[i][j]]+'\t'+str(similarity[i][j])+'\n')
                        fp_list.append(idx2phrase[i]+'\t'+idx2phrase[indices[i][j]]+'\t'+str(similarity[i][j])+'\n')
                    if i in indices[indices[i][j]]:
                        index = np.where(indices[indices[i][j]] == i)
                        similarity[indices[i][j]][index] = 0
    with open(args.output_dir+'fp.txt', 'w') as f:
        for string in random.sample(fp_list, 20):
            f.write(string)
    confusion_matrix[1, 0] += predicted_p - confusion_matrix[0, 0]
    print(confusion_matrix[1, 0] - fp)
    length = len(umls.stridx_list)
    confusion_matrix[1, 1] += (length * (length - 1) / 2 - confusion_matrix[0, 0] - confusion_matrix[0, 1] - confusion_matrix[1, 0])
    print('threshold:', threshold)
    print(confusion_matrix)
    return confusion_matrix, similarity_list_actual_p, similarity_list_actual_n

def print_result(threshold_list, accuracy_list, recall_list, precision_list, args):
    table = PrettyTable()
    column_names = ["Threshold", "Accuracy", "Recall", "Precision", "F1"]
    table.add_column(column_names[0], threshold_list)
    table.add_column(column_names[1], [format(accuracy, '.3f') for accuracy in accuracy_list])
    table.add_column(column_names[2], [format(recall, '.3f') for recall in recall_list])
    table.add_column(column_names[3], [format(precision, '.3f') for precision in precision_list])
    table.add_column(column_names[4], [format(2*precision*recall/(precision+recall), '.3f') for (precision, recall) in zip(precision_list, recall_list)])
    print(table)
    table = table.get_string()
    with open(args.output_dir+args.title+'.txt', 'w') as f:
        f.write(table)

def plot_histogram(listp, listn, name):
    plt.figure()
    plt.hist(listp, bins=50, range=(min(listn), 1), density=False, alpha=0.5, label='Pair with same Cui')
    plt.hist(listn, bins=50, range=(min(listn), 1), density=False, alpha=0.5, label='Pair with different Cui')
    plt.xlabel('Similarity score')
    plt.ylabel('Frequency')
    plt.title('Similarity score for pairs(Frequency)')
    plt.legend(loc='upper left')
    plt.savefig('frequency_' + name + '.png')
    plt.show()

    plt.figure()
    plt.hist(listp, bins=50, range=(min(listn), 1), density=True, alpha=0.5, label='Pair with same Cui')
    plt.hist(listn, bins=50, range=(min(listn), 1), density=True, alpha=0.5, label='Pair with different Cui')
    plt.xlabel('Similarity score')
    plt.ylabel('Density')
    plt.title('Similarity score for pairs(Density)')
    plt.legend(loc='upper left')
    plt.savefig('density_' + name + '.png')
    plt.show()    

def run(args):
    accuracy_list = []
    recall_list = []
    precision_list = []
    thre_list = []
    for threshold in threshold_list:
        similarity = np.load(args.similarity_path)
        indices = np.load(args.indices_path)
        confusion_matrix, similarity_list_actual_p, similarity_list_actual_n = eval_clustering(umls, similarity, indices, threshold, args)
        accuracy_list.append((confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum())
        recall_list.append(confusion_matrix[0, 0] / confusion_matrix[0].sum())
        precision_list.append(confusion_matrix[0, 0] / confusion_matrix[:, 0].sum())
        thre_list.append(threshold)
        print_result(thre_list, accuracy_list, recall_list, precision_list, args)
    # plot_histogram(similarity_list_actual_p, similarity_list_actual_n, args.title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umls_dir",
        default="/media/sda1/GanjinZero/UMLSBert/umls",
        type=str,
        help="Directory of umls data"
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_data_dir",
        default="use_data/",
        type=str,
        help="Directory of faiss index, idx2phrase and other use data"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title of the task"
    )
    args = parser.parse_args()
    
    args.indices_path = os.path.join(args.use_data_dir, "indices.npy")
    args.similarity_path = os.path.join(args.use_data_dir, "similarity.npy")
    args.phrase2idx_path = os.path.join(args.use_data_dir, "phrase2idx.pkl")
    args.idx2phrase_path = os.path.join(args.use_data_dir, "idx2phrase.pkl")
    umls = UMLS(umls_path=args.umls_dir, phrase2idx_path=args.phrase2idx_path, idx2phrase_path=args.idx2phrase_path)
    threshold_list = [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]
    # threshold_list = [.8]
    run(args)

