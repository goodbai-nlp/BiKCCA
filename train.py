#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: train.py
@time: 17-3-29 下午10:32
"""
import numpy as np
import time
from Kcca import KCCA
import utils
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-slang", "--src_lang", type=str, help="source language ")
parser.add_argument("-tlang", "--tgt_lang", type=str, help="target language ")
parser.add_argument("-semb", "--src", type=str, help="source language embedding path")
parser.add_argument("-temb", "--tgt", type=str, help="target language embedding path")
parser.add_argument("-d", "--dict", type=str, help="training dict path")
parser.add_argument("-reg", "--reg", type=float, help="regulation term of KCCA")
parser.add_argument("-g1", "--gamma1", type=float, help="gamma used in source Gussian kernel")
parser.add_argument("-g2", "--gamma2", type=float, help="gamma used in target Gussian kernel")


def KCCA_project_vectors(args,src_dico,tgt_dico,src_full,tgt_full,src_train,tgt_train,NUM_dim=100):

    kcca = KCCA('rbf', 'rbf', regularization=args.reg, gamma1=args.gamma1, gamma2=args.gamma2, n_jobs=-1,
                n_components=NUM_dim)
    kcca.fit(src_train, tgt_train)
    print('Finish training')

    print('Transforming embeddings...')
    src_projected,tgt_projected = kcca.transform(src_full,tgt_full)
    src_out,tgt_out = utils.norm_embeddings(src_projected),utils.norm_embeddings(tgt_projected)

    print('Exporting embeddings...')
    OutputDir = "output/{}-{}/".format(args.src_lang,args.tgt_lang)
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    utils.export_embeddings(src_dico[0],src_out,OutputDir+'projected.{}'.format(args.src_lang))
    utils.export_embeddings(tgt_dico[0],tgt_out,OutputDir+'projected.{}'.format(args.tgt_lang))

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

        if len(src_train)==7000:  # 7000 seeds for training
            break

    assert len(src_train) == len(tgt_train)

    src_vec = np.array(src_train)
    tgt_vec = np.array(tgt_train)
    return src_vec,tgt_vec



def Predeal_dicts(src_wc_path,tgt_wc_path,dict_path,threthold):
    src_wc, tgt_wc = utils.load_word_count(src_wc_path), utils.load_word_count(tgt_wc_path)
    print('Src_wc:{} tgt_wc:{}'.format(len(src_wc), len(tgt_wc)))
    utils.filter_dict(dict_path, src_wc, tgt_wc, threthold=threthold)

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
    KCCA_project_vectors(args, src_dico, tgt_dico, src_full, tgt_full, src_train, tgt_train, NUM_dim=100)
    print('Finished')

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
