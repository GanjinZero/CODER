import numpy as np
from tqdm import tqdm
import pickle
import argparse

# reference: https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm
class DisjointSet(object):
    def __init__(self):
        self.leader = dict()     # maps a member to the group's leader
        self.group = dict()  # maps a group leader to the group (which is a set)
    
    def add(self, a, b):
        leadera = self.leader.get(a)   
        leaderb = self.leader.get(b)   
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

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

    indices = np.load(args.indices_path)
    similarity = np.load(args.similarity_path)
    with open(args.idx2phrase_path, 'rb') as f:
        idx2phrase = pickle.load(f)
    
    ds = DisjointSet()
    for idxi in tqdm(range(indices.shape[0])):
        a = idx2phrase[idxi]
        if len(a) <= 3:
            continue
        for idxj in range(indices.shape[1]):
            if similarity[idxi, idxj] > 0.8:
                b = idx2phrase[indices[idxi][idxj]]
                if len(b) <= 3:
                    continue
                ds.add(a, b)
    
    with open(args.result_path, 'wb') as f:
        pickle.dump(ds.group, f)
    print('done')
