#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: utils.py
@time: 2017/3/24 16:22
"""
import io
import numpy as np
def load_embeddings(emb_path, vocab=None):
    word2id = {}
    vectors = []
    _emb_dim_file = 200
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                splits = line.split()
                assert len(splits) == 2
                _emb_dim_file = int(splits[1])
                print('Num. embeddings:{},Dim:{}'.format(splits[0],splits[1]))
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                if vocab and not word in vocab:
                    continue
                else:
                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word in word2id:
                        print("Word '%s' found twice in embedding file" % (word))
                    else:
                        if not vect.shape[0] == _emb_dim_file:
                            print("Invalid dimension %i for word '%s' in line %i." % (vect.shape[0], word, i))
                            continue
                        assert vect.shape == (_emb_dim_file,), i
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = (id2word, word2id)
    embeddings = np.concatenate(vectors, 0)
    return dico, embeddings

def center_embeddings(emb):
    print('Centering the embeddings')
    mean = emb.mean(0)
    emb = emb-mean
    return emb

def norm_embeddings(emb):
    print('Normalizing the embeddings')
    norms = np.linalg.norm(emb,axis=1,keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    return emb

def export_embeddings(dico, emb, path, eformat='txt'):
    assert eformat in ["txt", "pth"]
    if eformat == "txt":
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % emb.shape)
            for i in range(len(dico)):
                f.write(u"%s %s\n" % (dico[i], " ".join('%.5f' % x for x in emb[i])))
    # if eformat == "pth":
    #     torch.save({'dico':dico, 'vectors':emb}, path)
    else:
        print('Invalid format for export!')

def load_word_count(path):
    res = {}
    gold_f = open(path, encoding='utf-8', errors='surrogateescape')
    for line in gold_f:
        try:
            key, value = line.strip().split(':')
            res[key] = int(value)
        except ValueError:
            continue
    return res

def load_dict(path):
    word_src,word_tgt = [],[]
    gold_f = open(path, encoding='utf-8', errors='surrogateescape')
    for line in gold_f:
        src, trg = line.strip().split(' ||| ')
        word_src.append(src)
        word_tgt.append(trg)
    return (word_src,word_tgt)

def filter_dict(dpath,src_wc,tgt_wc,threthold):
    filtered = []
    gold_f = open(dpath, encoding='utf-8', errors='surrogateescape')
    for line in gold_f:
        trg, src = line.strip().split(' ||| ')
        if src in src_wc and trg in tgt_wc:
            if src_wc[src]>threthold and tgt_wc[trg]>threthold:
                tmpstr = '{} ||| {}'.format(trg,src)
                filtered.append(tmpstr)
    out_f = open(dpath+'.new','w+',encoding='utf-8')
    print('Filted dicts: {}'.format(len(filtered)))
    out_f.write('\n'.join(filtered))