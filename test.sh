name=test_allSep3..6..04.120.pred

python src/tags2sst.py streusle-2.0/split/$name.tags > streusle-2.0/split/$name.sst
python src/mweval.py streusle-2.0/split/streusle.test_all.sst streusle-2.0/split/$name.sst
python src/ssteval.py streusle-2.0/split/streusle.test_all.sst streusle-2.0/split/$name.sst