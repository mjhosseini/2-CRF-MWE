#!/bin/bash
set -eu
type=3
numIter=80
itersMod=40
cutoff=3
#cp src/classifiers.py src/classifiers2.pyx
#cp src/classifiersD.py src/classifiersD2.pyx

C=C3-reverse

train=../dimsum-data-1.5/split/dimsum16.train_allV
test=../dimsum-data-1.5/split/dimsum16.train_allT

alpha=2.4

for alpha2 in `seq 0.08 .04 .32`; do
echo $alpha2

outname=test$C$type-$alpha-$alpha2-$numIter-$cutoff.pred
outname=../dimsum-data-1.5/split/tuneD2/$outname

unbuffer python2.7 ../src/main.py --iters $numIter --itersMod $itersMod --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2 >"tuneD2/log$C$alpha2.txt" &

done

wait

for alpha2 in `seq 0.08 .04 .32`; do

outname=test$C$type-$alpha-$alpha2-$numIter-$cutoff.pred
outname=../dimsum-data-1.5/split/tuneD2/$outname

python2.7 ../src/streusle2dimsum.py $outname.tags $outname
python2.7 ../dimsum-data-1.5/scripts/dimsumeval.py $test $outname >> "tuneD2/log$C$alpha2.txt"

done
