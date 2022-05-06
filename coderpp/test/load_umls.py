import os
from tqdm import tqdm
import re
from random import shuffle
import pickle
import ahocorasick
#import ipdb

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return


class UMLS(object):
    def __init__(self, umls_path, phrase2idx_path, idx2phrase_path, source_range=None, lang_range=['ENG'], only_load_dict=False):
        # phrase2idx is the dict of our NER vocab. It is used to exclude those phrases in MRCONSO but not in our NER vocab
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.phrase2idx = self._load_pickle(phrase2idx_path)
        self.idx2phrase = self._load_pickle(idx2phrase_path)
        self.detect_type()
        self.load()

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def transform(self, phrase):
        if phrase in self.phrase2idx.keys() and len(phrase)>3:
            return self.phrase2idx[phrase]
        else:
            return None

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"

    def load(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        self.lui_set = set()
        self.cui2str = {}
        self.str2cui = {}
        self.code2cui = {}
        self.stridx_list = set()
        #self.lui_status = {}
        read_count = 0
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            # lui_status = l[2].lower() # p -> preferred
            lui = l[3]
            source = l[11]
            code = l[13]
            string = l[14]

            if (self.source_range is None or source in self.source_range) and (self.lang_range is None or lang in self.lang_range):
                if not lui in self.lui_set:
                    clean_string = self.clean(string)
                    idx = self.transform(clean_string)
                    if idx is None:
                        continue
                    read_count += 1
                    # if 'abdom' not in clean_string:
                    #     continue
                    if string not in self.str2cui:
                        self.str2cui[string] = set()
                    self.str2cui[string].update([cui])
                    if string.lower() not in self.str2cui:
                        self.str2cui[string.lower()] = set()
                    self.str2cui[string.lower()].update([cui])
                    if clean_string not in self.str2cui:
                        self.str2cui[clean_string] = set()
                    self.str2cui[clean_string].update([cui])

                    if not cui in self.cui2str:
                        self.cui2str[cui] = set()
                    self.cui2str[cui].update([idx])
                    self.stridx_list.update([idx])
                    self.code2cui[code] = cui
                    self.lui_set.update([lui])

            # For debug
            # if len(self.stridx_list) > 500:
            #     break

        self.cui = list(self.cui2str.keys())
        shuffle(self.cui)
        self.cui_count = len(self.cui)
        self.stridx_list = list(self.stridx_list)

        print("cui count:", self.cui_count)
        print("str2cui count:", len(self.str2cui))
        print("MRCONSO count:", read_count)
        print("str count:", len(self.stridx_list))
        # print([[self.idx2phrase[stridx] for stridx in list(gt_clustering)] for gt_clustering in list(self.cui2str.values())])

    def clean(self, term, lower=True, clean_NOS=True, clean_bracket=True, clean_dash=True):
        term = " " + term + " "
        if lower:
            term = term.lower()
        if clean_NOS:
            term = term.replace(" NOS ", " ").replace(" nos ", " ")
        if clean_bracket:
            term = re.sub(u"\\(.*?\\)", "", term)
        if clean_dash:
            term = term.replace("-", " ")
        term = " ".join([w for w in term.split() if w])
        return term

