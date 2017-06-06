#!/bin/bash
set -eu
type=2
#cp src/classifiers.py src/classifiers2.pyx
#cp src/classifiersD.py src/classifiersD2.pyx

alpha=.6

for alpha2 in `seq 0.02 .02 .08`; do
echo $alpha2
unbuffer python2.7 src/main.py --iters 4 --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_allT.tags --test-predict streusle-2.0/split/streusle.test_allV.tags --outFilePath streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha --alpha2 $alpha2 >"tune2/log$alpha2.txt" &
#python2.7 src/main.py --iters 4 --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_allT.tags --test-predict streusle-2.0/split/streusle.test_allV.tags --outFilePath streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2 1>"tune2/log$alpha2.txt" &



done

wait

for alpha2 in `seq 0.02 .02 .08`; do

python src/tags2sst.py streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.tags 1>streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.sst
python src/mweval.py streusle-2.0/split/streusle.test_allV.sst streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.sst >>"tune2/log$alpha2.txt"
python src/ssteval.py streusle-2.0/split/streusle.test_allV.sst streusle-2.0/split/tune2/test_allV$type.pred.$alpha2.sst >>"tune2/log$alpha2.txt"

done
