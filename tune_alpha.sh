#!/bin/bash
set -eu
type=2
#cp src/classifiers.py src/classifiers2.pyx
#cp src/classifiersD.py src/classifiersD2.pyx


for alpha in `seq 0 .3 3`; do
echo $alpha
python2.7 src/main.py --iters 4 --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_allT.tags --test-predict streusle-2.0/split/streusle.test_allV.tags --outFilePath streusle-2.0/split/tune/test_allV$type.pred.$alpha.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha >"tune/log$alpha.txt" &
#python2.7 src/main.py --iters 4 --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_allT.tags --test-predict streusle-2.0/split/streusle.test_allV.tags --outFilePath streusle-2.0/split/tune/test_allV$type.pred.$alpha.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha 1>"tune/log$alpha.txt" &

	

done

wait

for alpha in `seq 0 .3 3`; do

python src/tags2sst.py streusle-2.0/split/tune/test_allV$type.pred.$alpha.tags 1>streusle-2.0/split/tune/test_allV$type.pred.$alpha.sst
python src/mweval.py streusle-2.0/split/streusle.test_allV.sst streusle-2.0/split/tune/test_allV$type.pred.$alpha.sst >>"tune/log$alpha.txt"
python src/ssteval.py streusle-2.0/split/streusle.test_allV.sst streusle-2.0/split/tune/test_allV$type.pred.$alpha.sst >>"tune/log$alpha.txt"

done
