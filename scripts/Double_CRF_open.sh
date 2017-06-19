set -eu
#Double CRF
type=3
alpha=1.6
alpha2=.2
numIter=120
#Saving outputs periodically after $itersMod iters
itersMod=40
cutoff=3

outname=../dimsum-data-1.5/split/out$type-$alpha-$alpha2-$numIter-$cutoff.pred

train=../dimsum-data-1.5/dimsum16.train
test=../dimsum-data-1.5/dimsum16.test.gold

#In case you're interested in the validation test in the paper:
#train=../dimsum-data-1.5/split/dimsum16.train_allV
#test=../dimsum-data-1.5/split/dimsum16.train_allT


#Train and test for open condition:
python2.7 ../src/main.py --iters $numIter --itersMod $itersMod --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --lex ../mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha --alpha2 $alpha2

#Train and test for closed condition:
#python2.7 ../src/main.py --iters $numIter --itersMod $itersMod --cutoff $cutoff --YY ../tagsets/bio2gNV_dim --defaultY O --debug --train $train.tags --test-predict $test.tags --outFilePath $outname.tags --bio NO_SINGLETON_B --cluster-file ../mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2

#The next two lines is for evaluation of the predicted tags
python2.7 ../src/streusle2dimsum.py $outname.tags $outname
python2.7 ../dimsum-data-1.5/scripts/dimsumeval.py $test $outname
