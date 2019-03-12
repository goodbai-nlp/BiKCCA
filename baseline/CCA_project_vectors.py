#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: project_vectors.py
@time: 17-3-29 上午9:28
"""
import numpy as np
import time
import argparse
from sklearn.cross_decomposition import CCA
import sys
sys.path.append("..")
import utils
import os

parser = argparse.ArgumentParser()

parser.add_argument("-slang", "--src_lang", type=str, help="source language ")
parser.add_argument("-tlang", "--tgt_lang", type=str, help="target language ")
parser.add_argument("-semb", "--src", type=str, help="source language embedding path")
parser.add_argument("-temb", "--tgt", type=str, help="target language embedding path")
parser.add_argument("-d", "--dict", type=str, help="training dict path")

def GetTrain(src_dico, src_full,tgt_dico, tgt_full,tgt_word,src_word):
    src_train,tgt_train = [],[]
    assert len(src_word)==len(tgt_word)

    for i in range(len(src_word)):
        src_w,tgt_w = src_word[i],tgt_word[i]
        if src_w in src_dico[1] and tgt_w in tgt_dico[1]:
            # print(src_w,tgt_w)
            src_idx, tgt_idx = src_dico[1][src_word[i]], tgt_dico[1][tgt_word[i]]
            src_train.append(src_full[src_idx])
            tgt_train.append(tgt_full[tgt_idx])

        if len(src_train)==500:
            break

    assert len(src_train) == len(tgt_train)

    src_vec = np.array(src_train)
    tgt_vec = np.array(tgt_train)
    return src_vec,tgt_vec

def CCA_project_vectors(args, src_dico, tgt_dico, src_full, tgt_full, src_train, tgt_train, NUM_dim=100):

    print('Exporting embeddings...')
    OutputDir = "output/{}-{}/".format(args.src_lang, args.tgt_lang)
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    cca = CCA(n_components=NUM_dim)
    print("Fitting...")
    cca.fit(src_train, tgt_train)
    print(cca.get_params())
    X_c, Y_c = cca.transform(src_full, tgt_full)
    src_out, tgt_out = utils.norm_embeddings(X_c), utils.norm_embeddings(Y_c)
    print("Exporting embeddings...")
    utils.export_embeddings(src_dico[0], src_out, OutputDir + 'projected.{}'.format(args.src_lang))
    utils.export_embeddings(tgt_dico[0], tgt_out, OutputDir + 'projected.{}'.format(args.tgt_lang))
    print("work over!")

if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    args = parser.parse_args()

    print('Loading training data...')
    tgt_word, src_word = utils.load_dict(args.dict)
    src_dico, src_full = utils.load_embeddings(args.src, vocab=None)
    tgt_dico, tgt_full = utils.load_embeddings(args.tgt, vocab=None)
    src_full = utils.norm_embeddings(src_full)
    tgt_full = utils.norm_embeddings(tgt_full)
    # print(src_full.shape,tgt_full.shape)
    src_train, tgt_train = GetTrain(src_dico, src_full, tgt_dico, tgt_full, tgt_word, src_word)
    print('Train dataset size:{}'.format(len(src_train)))

    print('Start training model...')
    CCA_project_vectors(args, src_dico, tgt_dico, src_full, tgt_full, src_train, tgt_train, NUM_dim=100)

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
