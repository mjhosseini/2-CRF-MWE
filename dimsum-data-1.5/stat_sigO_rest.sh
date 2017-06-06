set -eu

src='./'
src2="split/comp/open/"

###Valid

addr1="testFinalO-valid3-1.6-.2-130-3.pred-curCount=80"
addr2="testFinalO-valid-0-1.2-.12-10-3.pred"
addrGold="split/dimsum16.train_allT"

#unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_open_CRFD80.txt" &

addr1="testFinalO-valid-2-1.2-.12-130-3.pred-curCount=120"
addr2="testFinalO-valid-0-1.2-.12-10-3.pred"
addrGold="split/dimsum16.train_allT"

#unbuffer python2.7 ../Util/statSignificaneTest.py $src $src2$addr1 $src2$addr2 $addrGold 1>"out/valid_open_CRF120.txt" &




