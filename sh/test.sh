outname=test2..6..04.10.pred
python2.7 ../src/streusle2dimsum.py ../dimsum-data-1.5/split/$outname.tags ../dimsum-data-1.5/split/$outname

python2.7 ../dimsum-data-1.5/scripts/dimsumeval.py ../dimsum-data-1.5/split/dimsum16.train_allV ../dimsum-data-1.5/split/$outname