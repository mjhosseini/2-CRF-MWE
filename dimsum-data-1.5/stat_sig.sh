set -eu

src='./'
src2="split/comp/closed/"

###DIM

addr1="testFinalC-DIMG-Flush-3-1.6-.12-240-3.pred-curCount=120"
addr2="testFinalC-dim0-1.2-.32-10-3.pred"
addrGold="dimsum16.test.gold"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_closed_CRFD120.txt" &

###Valid

addr1="testFinalC-valid3-1.6-.12-130-3.pred-curCount=120"
addr2="testFinalC-valid0-1.6-.32-10-3.pred"
addrGold="split/dimsum16.train_allT"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_closed_CRFD120.txt" &


