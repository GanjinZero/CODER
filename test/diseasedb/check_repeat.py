from tqdm import tqdm
import os


def load_all_relation():
    with open("./data/relation_all.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    rel_list = []
    count = 0
    for line in tqdm(lines):
        cui1, rel, cui2, source = line.strip().split("|")
        if source == "diseasedb":
            count += 1
            rel = cui1 + "|" + cui2
            rel_list.append(rel)

    print("Tri group count:", count)

    return rel_list

def load_mrrel():
    with open("../../umls/MRREL.RRF", "r", encoding="utf-8") as f:
        lines = f.readlines()
    rel_set = set()
    for line in tqdm(lines):
        l = line.split("|")
        cui1 = l[0]
        cui2 = l[4]
        rel = cui1 + "|" + cui2
        reverse_rel = cui2 + "|" + cui1
        rel_set.update([rel, reverse_rel])

    return rel_set

diseasedb_list = load_all_relation()
mrrel_set = load_mrrel()
repeat = 0
for rel in tqdm(diseasedb_list):
    if rel in mrrel_set:
        repeat += 1
print(repeat, len(diseasedb_list))