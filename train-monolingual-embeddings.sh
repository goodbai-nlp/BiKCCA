# export output_embeddings="new_location" #"/usr1/home/wammar/cca-embeddings/all_languages.cca.window_5+iter_10+size_40+threads_16"
export temp="/home/xfbai/tmpvec"
export utils="/home/xfbai/mywork/git/KCCA-Experiment/utils"
export word2vec="/home/xfbai/tools/word2vec"

# create temp dir
mkdir $temp

# remove old embeddings if any
# rm $output_embeddings
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`

export corpus_en="/home/xfbai/corpus/monolingual/mono.tok.lc.en"

$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 20 -threads 16 -output $temp/embeddings_size20.en
#python $utils/Count.py -w1 $corpus_en -o $temp/en_wordCount.txt
#python $utils/Predeal.py -w1 $temp/embeddings_size100.en -w2 $temp/en_wordCount.txt -o $temp/new_embedding_size100.en

export corpus_zh="/home/xfbai/corpus/monolingual/mono.tok.lc.zh"
$word2vec/word2vec -train $corpus_zh -window 5 -iter 10 -size 20 -threads 16 -output $temp/embeddings_size20.zh
#python $utils/Count.py -w1 $corpus_zh -o $temp/zh_wordCount.txt
#python $utils/Predeal.py -w1 $temp/embeddings_size100.zh -w2 $temp/zh_wordCount.txt -o $temp/new_embedding_size100.zh
:<<!
# process en-de
#export corpus_en="/home/xfbai/corpus/monolingual/mono.tok.en"
export corpus_en="/home/xfbai/corpus/monolingual/mono.tok.lc.en"
$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.en
python $utils/Count.py -w1 $corpus_en -o $temp/en_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.en -w2 $temp/en_wordCount.txt -o $temp/new_embedding_size200.en
#!
:<<!
export corpus_de="/home/xfbai/corpus/monolingual/mono.tok.lc.de"
# export corpus_de="/home/xfbai/corpus/monolingual/mono.tok.de"
$word2vec/word2vec -train $corpus_de -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.de
python $utils/Count.py -w1 $corpus_de -o $temp/de_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.de -w2 $temp/de_wordCount.txt -o $temp/new_embedding_size200.de
# process fr
export corpus_fr="/home/xfbai/corpus/monolingual/mono.tok.lc.fr"
$word2vec/word2vec -train $corpus_fr -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.fr
python $utils/Count.py -w1 $corpus_fr -o $temp/fr_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.fr -w2 $temp/fr_wordCount.txt -o $temp/new_embedding_size200.fr
:<<!
export corpus_zh="/home/xfbai/corpus/monolingual/mono.tok.lc.zh"
$word2vec/word2vec -train $corpus_zh -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.zh
python $utils/Count.py -w1 $corpus_zh -o $temp/zh_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.zh -w2 $temp/zh_wordCount.txt -o $temp/new_embedding_size200.zh

:<<!
export corpus_fi="/home/xfbai/corpus/monolingual/mono.tok.lc.fi"
$word2vec/word2vec -train $corpus_fi -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.fi
python $utils/Count.py -w1 $corpus_fi -o $temp/fi_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.fi -w2 $temp/fi_wordCount.txt -o $temp/new_embedding_size200.fi

export corpus_hu="/home/xfbai/corpus/monolingual/mono.tok.lc.hu"
$word2vec/word2vec -train $corpus_hu -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.hu
python $utils/Count.py -w1 $corpus_hu -o $temp/hu_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.hu -w2 $temp/hu_wordCount.txt -o $temp/new_embedding_size200.hu


export corpus_cs="/home/xfbai/corpus/monolingual/mono.tok.lc.cs"
$word2vec/word2vec -train $corpus_cs -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.cs
python $utils/Count.py -w1 $corpus_cs -o $temp/cs_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.cs -w2 $temp/cs_wordCount.txt -o $temp/new_embedding_size200.cs

export corpus_ar="/home/xfbai/corpus/monolingual/mono.tok.lc.ar"
$word2vec/word2vec -train $corpus_ar -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.ar
python $utils/Count.py -w1 $corpus_ar -o $temp/ar_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.ar -w2 $temp/ar_wordCount.txt -o $temp/new_embedding_size200.ar

export corpus_ru="/home/xfbai/corpus/monolingual/mono.tok.lc.ru"
$word2vec/word2vec -train $corpus_ru -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.ru
python $utils/Count.py -w1 $corpus_ru -o $temp/ru_wordCount.txt
python $utils/Predeal.py -w1 $temp/embeddings_size200.ru -w2 $temp/ru_wordCount.txt -o $temp/new_embedding_size200.ru
!
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
