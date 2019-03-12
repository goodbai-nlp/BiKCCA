# BiKCCA
Code for our paper "Improving Vector Space Word Representations Via Kernel Canonical Correlation Analysis" in TALLIP [[pdf]](https://dl.acm.org/citation.cfm?id=3197566)
## Setup
This software runs python 3.6 with the following libraries:
+ numpy 1.16.2
+ scikit-learn 0.20.2
## Get start
1. Preparing monolingual word embeddings and dictionaris.
```
$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 200 -threads 16 -output embeddings_size200.en 
$word2vec/word2vec -train $corpus_zh -window 5 -iter 10 -size 200 -threads 16 -output embeddings_size200.zh 
```

2. Generating bilingual word embeddings with our method (BiKCCA).
```
python train.py -slang $src_lang -tlang $tgt_lang -semb $src_path -temb $tgt_path -d $dict_path -reg 0.3  -g1 0.001  -g2 0.001
```
The `reg`, `g1` and `g2` are hyperparameters of KCCA, which can be tuned on valid dataset.

3. The resulted bilingual word embeddings will be stored at directory `output/src_lang-tgt_lang/`

4. To evaluate the bilingual word embeddings, please refer to the code of this [work](https://github.com/shyamupa/biling-survey)

## References
Please cite Learning [Improving Vector Space Word Representations Via Kernel Canonical Correlation Analysis](https://dl.acm.org/citation.cfm?id=3197566) if you found the resources in this repository useful.
```
  @article{Bai:2018:IVS:3229525.3197566,
   author = {Bai, Xuefeng and Cao, Hailong and Zhao, Tiejun},
   title = {Improving Vector Space Word Representations Via Kernel Canonical Correlation Analysis},
   journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
   issue_date = {August 2018},
   volume = {17},
   number = {4},
   month = jul,
   year = {2018},
   issn = {2375-4699},
   pages = {29:1--29:16},
   url = {http://doi.acm.org/10.1145/3197566},
   doi = {10.1145/3197566},
   acmid = {3197566},
   publisher = {ACM},
   address = {New York, NY, USA}
  } 
```
