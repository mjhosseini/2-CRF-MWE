set -eu
type=3
#alpha=1.6
#alpha2=.12
alpha=2.4
alpha2=.2
#numIter=130
numIter=5
itersMod=40
cutoff=3

C=C-valid
outname=testFinal$C$type-$alpha-$alpha2-$numIter-$cutoff.pred
#cp ../src/classifiers.py ../src/classifiers2.pyx
#cp ../src/classifiersD.py ../src/classifiersD2.pyx
#$trainP = /Users/hosseini/Desktop/D/research/MWE/data/dimsum-data-1.2/split/streusle.train.tags
#$testP = /Users/hosseini/Desktop/D/research/MWE/data/dimsum-data-1.2/split/streusle.test.tags

# prediction only

train=../dimsum-data-1.5/split/dimsum16.train_allV
test=../dimsum-data-1.5/split/dimsum16.train_allT

#train=../dimsum-data-1.5/split/dimsum16.train
#test=../dimsum-data-1.5/split/dimsum16.test.blind

outname=../dimsum-data-1.5/split/$outname

#python2.7 ../src/dimsum2streusle.py $train $train.tags
#python2.7 ../src/dimsum2streusle.py $test $test.tags



#python2.7 ../src/main.py --iters $numIter --itersMod 5 --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --lex ../mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha --alpha2 $alpha2
python2.7 ../src/main.py --iters $numIter --itersMod $itersMod --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2

python2.7 ../src/streusle2dimsum.py $outname.tags $outname

python2.7 ../dimsum-data-1.5/scripts/dimsumeval.py $test $outname

