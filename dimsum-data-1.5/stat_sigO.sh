set -eu

src='./'
src2="split/comp/open/"

###DIM

addr1="testFinalO-dim-1-2.4-.32-130-3.pred"
addr2="testFinalO-dim0-1.2-.32-10-3.pred"
addrGold="dimsum16.test.gold"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_open_MLR.txt" &

addr1="testFinalO-DIMG-3-1.6-.2-240-3.pred-curCount=120"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_open_CRFD120.txt" &

addr1="testFinalO-DIMG-3-1.6-.2-240-3.pred-curCount=80"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_open_CRFD80.txt" &


###Valid

addr1="testFinalO-valid3-1.6-.2-130-3.pred-curCount=120"
addr2="testFinalO-valid-0-1.2-.12-10-3.pred"
addrGold="split/dimsum16.train_allT"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_open_CRFD120.txt" &


