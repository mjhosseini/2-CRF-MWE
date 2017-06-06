set -eu

src='./'
src2="split/comp/open/"

###DIM

addr1="testFinalO-DIMG-3-1.6-.2-240-3.pred-curCount=120"
addr2="testFinalO-DIM-2-1.2-.12-480-3.pred-curCount=120"
addrGold="dimsum16.test.gold"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_open_CRF.txt" &


###Valid

addr1="testFinalO-valid3-1.6-.2-130-3.pred-curCount=120"
addr2="testFinalO-valid-2-1.2-.12-130-3.pred-curCount=120"
addrGold="split/dimsum16.train_allT"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_open_CRF.txt" &


######Closed

src2="split/comp/closed/"

###DIM

addr1="testFinalC-DIMG-Flush-3-1.6-.12-240-3.pred-curCount=120"
addr2="testFinalC-DIM-2-1.6-.32-480-3.pred-curCount=120"
addrGold="dimsum16.test.gold"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_closed_CRF.txt" &


###Valid

addr1="testFinalC-valid3-1.6-.12-130-3.pred-curCount=120"
addr2="testFinalC-valid2-1.6-.32-130-3.pred-curCount=120"
addrGold="split/dimsum16.train_allT"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_closed_CRF.txt" &