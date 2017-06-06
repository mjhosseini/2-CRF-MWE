set -eu
type=2
alpha=.6
alpha2=.04
numIter=120
outname=test_allOpt_F$type.$alpha.$alpha2.$numIter.pred
cp src/classifiers.py src/classifiers2.pyx
cp src/classifiersD.py src/classifiersD2.pyx
#$trainP = /Users/hosseini/Desktop/D/research/MWE/data/dimsum-data-1.2/split/streusle.train.tags
#$testP = /Users/hosseini/Desktop/D/research/MWE/data/dimsum-data-1.2/split/streusle.test.tags

# prediction only
	

	#python2.7 src/main.py --iters $numIter --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_all.tags --test-predict streusle-2.0/split/streusle.test_all.tags --outFilePath streusle-2.0/split/$outname.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --ctype $type --alpha $alpha --alpha2 $alpha2
	python2.7 src/main.py --iters 4 --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --train streusle-2.0/split/streusle.train_all.tags --test-predict streusle-2.0/split/streusle.test_all.tags --outFilePath streusle-2.0/split/$outname.tags --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --ctype $type --alpha $alpha --alpha2 $alpha2
	
	
	python src/tags2sst.py streusle-2.0/split/$outname.tags > streusle-2.0/split/$outname.sst
	python src/mweval.py streusle-2.0/split/streusle.test_all.sst streusle-2.0/split/$outname.sst
	python src/ssteval.py streusle-2.0/split/streusle.test_all.sst streusle-2.0/split/$outname.sst
	
	#python2.7 src/dimsumeval.py /Users/hosseini/Desktop/D/research/MWE/data/streusle-2.1/split/test.gold /Users/hosseini/Desktop/D/research/MWE/data/streusle-2.1/split/testt$type.pred
	#python2.7 src/dimsumeval.py /Users/hosseini/Desktop/D/research/MWE/data/streusle-2.0/split/test_all.gold /Users/hosseini/Desktop/D/research/MWE/data/streusle-2.0/split/test_all$type.pred
	
