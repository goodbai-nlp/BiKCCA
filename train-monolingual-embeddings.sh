export output="/output/path/"
export word2vec="/path/to/word2vec"

# create output dir
mkdir $output

printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`

export corpus_en="/path/to/corpus/mono.tok.lc.en"
$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 200 -threads 16 -output $output/embeddings_size200.en

export corpus_zh="/path/to/corpus/mono.tok.lc.zh"
$word2vec/word2vec -train $corpus_zh -window 5 -iter 10 -size 200 -threads 16 -output $temp/embeddings_size200.zh


printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
