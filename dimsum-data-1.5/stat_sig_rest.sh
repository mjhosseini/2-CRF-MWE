set -eu

src='./'
src2="split/comp/closed/"

###DIM

addr1="testFinalC-DIM-2-1.6-.32-480-3.pred-curCount=120"
addr2="testFinalC-dim0-1.2-.32-10-3.pred"
addrGold="dimsum16.test.gold"

#unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_closed_CRF.txt" &

###Valid

addr1="testFinalC-valid2-1.6-.32-130-3.pred-curCount=120"
addr2="testFinalC-valid0-1.6-.32-10-3.pred"
addrGold="split/dimsum16.train_allT"

#unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_closed_CRF.txt" &


addr1="testFinalC-validaa1-1.6-.32-130-3.pred"
addr2="testFinalC-dim0-1.2-.32-10-3.pred"
addrGold="dimsum16.test.gold"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/dim_closed_MLR.txt" &