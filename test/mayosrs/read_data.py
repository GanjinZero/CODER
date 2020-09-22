import os
import sys
from collections import defaultdict
import glob
import datetime
import re
import random


thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(thisdir, 'data')


def main():
    srs = get_srs()
    print(srs)


def get_srs():
    srs = _get_srs()
    srs_mini = _get_srs_mini()
    full = {}
    full.update(srs)
    full.update(srs_mini)
    return full


def get_srs_cui():
    srs = _get_srs_cui()
    srs_mini = _get_srs_mini_cui()
    full = {}
    full.update(srs)
    full.update(srs_mini)
    return full


def _get_srs():
    X, Y = [], []
    filename = os.path.join(datadir, 'MayoSRS.csv')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            toks = line.strip().split(',')
            score = float(toks[0])
            phrase1 = toks[3].strip('"').lower()
            phrase2 = toks[4].strip('"').lower()
            X.append((phrase1, phrase2))
            Y.append(score)
    scores = {'srs': (X, Y)}
    return scores


def _get_srs_cui():
    X, Y = [], []
    filename = os.path.join(datadir, 'MayoSRS.csv')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            toks = line.strip().split(',')
            score = float(toks[0])
            cui1 = toks[1].strip('"')
            cui2 = toks[2].strip('"')
            X.append((cui1, cui2))
            Y.append(score)
    scores = {'srs': (X, Y)}
    return scores


def _get_srs_mini():
    X, physician_Y, coder_Y = [], [], []
    filename = os.path.join(datadir, 'MiniMayoSRS.csv')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            toks = line.strip().split(',')
            physician_score = float(toks[0])
            coder_score = float(toks[1])
            phrase1 = toks[4].strip('"').lower()
            phrase2 = toks[5].strip('"').lower()
            X.append((phrase1, phrase2))
            physician_Y.append(physician_score)
            coder_Y.append(coder_score)
    scores = {'mini_physician': (X, physician_Y), 'mini_coder': (X, coder_Y)}
    return scores


def _get_srs_mini_cui():
    X, physician_Y, coder_Y = [], [], []
    filename = os.path.join(datadir, 'MiniMayoSRS.csv')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            toks = line.strip().split(',')
            physician_score = float(toks[0])
            coder_score = float(toks[1])
            cui1 = toks[2].strip('"')
            cui2 = toks[3].strip('"')
            X.append((cui1, cui2))
            physician_Y.append(physician_score)
            coder_Y.append(coder_score)
    scores = {'mini_physician': (X, physician_Y), 'mini_coder': (X, coder_Y)}
    return scores


if __name__ == '__main__':
    main()
    print(get_srs_cui())
