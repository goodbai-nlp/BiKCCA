#!/usr/bin/env bash
src_lang=en
tgt_lang=zh
datadir=/data/embeddings
src_path=$datadir/new_embedding_size200.$src_lang
tgt_path=$datadir/new_embedding_size200.$tgt_lang

dict_path=/data/dictionaries/ACL16/dict.$tgt_lang-$src_lang.new

python train.py -slang $src_lang -tlang $tgt_lang -semb $src_path -temb $tgt_path -d $dict_path -reg 0.3  -g1 0.001  -g2 0.001
