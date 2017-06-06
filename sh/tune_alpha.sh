#!/bin/bash
set -eu
type=1
numIter=80
itersMod=40
cutoff=3
#O means Open, C means Closed
C=C3-reverseee

#train=../dimsum-data-1.5/split/dimsum16.train_allV
#test=../dimsum-data-1.5/split/dimsum16.train_allT

train=../dimsum-data-1.5/split/dimsum16.train_allT
test=../dimsum-data-1.5/split/dimsum16.train_allV

#cp src/classifiers.py src/classifiers2.pyx
#cp src/classifiersD.py src/classifiersD2.pyx


for alpha in `seq .4 .4 3`; do
echo $alpha
alpha2=$alpha

outname=test$C$type-$alpha-$alpha2-$numIter-$cutoff.pred
outname=../dimsum-data-1.5/split/tune/$outname

#python2.7 ../src/main.py --iters $numIter --cutoff 5 --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --lex ../mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha --alpha2 $alpha2 >"tune/log$C$alpha.txt" &
unbuffer python2.7 ../src/main.py --iters $numIter --itersMod $itersMod --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2 >"tune/log$C-$type-$alpha.txt" &


done

wait

for alpha in `seq .4 .4 3`; do
alpha2=$alpha
outname=test$C$type-$alpha-$alpha2-$numIter-$cutoff.pred
outname=../dimsum-data-1.5/split/tune/$outname

python2.7 ../src/streusle2dimsum.py $outname.tags $outname
python2.7 ../dimsum-data-1.5/scripts/dimsumeval.py $test $outname >> "tune/log$C-$type-$alpha.txt"

done
