# Simply read umls and generate idx2phrase, phrase2idx
import pickle
import argparse
import os
from tqdm import tqdm
import re
from random import shuffle
#import ipdb

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return

class AllUMLS(object):
    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=True):
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()
        self.load()
        if not only_load_dict:
            self.load_rel()
            self.load_sty()

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
                    read_count += 1
                    # self.str2cui[string] = cui
                    # self.str2cui[string.lower()] = cui
                    clean_string = self.clean(string)
                    self.str2cui[clean_string] = cui

                    if not cui in self.cui2str:
                        self.cui2str[cui] = set()
                    self.cui2str[cui].update([clean_string])
                    self.code2cui[code] = cui
                    self.lui_set.update([lui])

            # For debug
            # if read_count > 1000:
            #     break

        self.cui = list(self.cui2str.keys())
        self.cui_count = len(self.cui)

        print("cui count:", self.cui_count)
        print("str2cui count:", len(self.str2cui))
        print("MRCONSO count:", read_count)

    def load_rel(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRREL." + self.type))
        self.rel = set()
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui0 = l[0]
            re = l[3]
            cui1 = l[4]
            rel = l[7]
            if cui0 in self.cui2str and cui1 in self.cui2str:
                str_rel = "\t".join([cui0, cui1, re, rel])
                if not str_rel in self.rel and cui0 != cui1:
                    self.rel.update([str_rel])

            # For debug
            # if len(self.rel) > 1000:
            #     break
        self.rel = list(self.rel)

        print("rel count:", len(self.rel))

    def load_sty(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        self.cui2sty = {}
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            sty = l[3]
            if cui in self.cui2str:
                self.cui2sty[cui] = sty

        print("sty count:", len(self.cui2sty))

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

    def search_by_code(self, code):
        if code in self.cui2str:
            return list(self.cui2str[code])
        if code in self.code2cui:
            return list(self.cui2str[self.code2cui[code]])
        return None

    def search_by_string_list(self, string_list):
        for string in string_list:
            if string in self.str2cui:
                find_string = self.cui2str[self.str2cui[string]]
                return [string for string in find_string if not string in string_list]
            if string.lower() in self.str2cui:
                find_string = self.cui2str[self.str2cui[string.lower()]]
                return [string for string in find_string if not string in string_list]
        return None

    def search(self, code=None, string_list=None, max_number=-1):
        result_by_code = self.search_by_code(code)
        if result_by_code is not None:
            if max_number > 0:
                return result_by_code[0:min(len(result_by_code), max_number)]
            return result_by_code
        return None
        result_by_string = self.search_by_string_list(string_list)
        if result_by_string is not None:
            if max_number > 0:
                return result_by_string[0:min(len(result_by_string), max_number)]
            return result_by_string
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umls_dir",
        default="../umls",
        type=str,
        help="UMLS dir",
    )
    parser.add_argument(
        "--use_data_dir",
        default="data",
        type=str,
        help="output dir for data like idx2phrase and phrase2idx"
    )
    args = parser.parse_args()

    umls = AllUMLS(
        umls_path=args.umls_dir
    )
    phrase_list = list(umls.str2cui.keys())

    idx2phrase = {idx:phrase for idx, phrase in enumerate(phrase_list)}
    print(type(idx2phrase))
    phrase2idx = {phrase:idx for idx, phrase in enumerate(phrase_list)}
    with open(os.path.join(args.use_data_dir, 'idx2phrase.pkl'), 'wb') as f:
        pickle.dump(idx2phrase, f)
    with open(os.path.join(args.use_data_dir, 'phrase2idx.pkl'), 'wb') as f:
        pickle.dump(phrase2idx, f)