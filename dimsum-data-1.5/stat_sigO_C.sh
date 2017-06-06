set -eu

src='./'
srco="split/comp/open/"
srcc="split/comp/closed/"



###DIM

#CRF
addrGold="dimsum16.test.gold"

addr1="testFinalO-DIM-2-1.2-.12-480-3.pred-curCount=120"
addr2="testFinalC-DIM-2-1.6-.32-480-3.pred-curCount=120"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $srco$addr1 $srcc$addr2 $addrGold 1>"out/dim_O_C_CRF.txt" &

#CRFD

addr1="testFinalO-DIMG-3-1.6-.2-240-3.pred-curCount=120"
addr2="testFinalC-DIMG-Flush-3-1.6-.12-240-3.pred-curCount=120"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $srco$addr1 $srcc$addr2 $addrGold 1>"out/dim_O_C_CRFD.txt" &


###Valid
#CRF

addrGold="split/dimsum16.train_allT"

addr1="testFinalO-valid-2-1.2-.12-130-3.pred-curCount=120"
addr2="testFinalC-valid2-1.6-.32-130-3.pred-curCount=120"


unbuffer python2.7 ../Util/statSignificaneTest.py $src $srco$addr1 $srcc$addr2 $addrGold 1>"out/valid_O_C_CRF.txt" &

#CRFD

addr1="testFinalO-valid3-1.6-.2-130-3.pred-curCount=120"
addr2="testFinalC-valid3-1.6-.12-130-3.pred-curCount=120"

unbuffer python2.7 ../Util/statSignificaneTest.py $src $srco$addr1 $srcc$addr2 $addrGold 1>"out/valid_O_C_CRFD.txt" &
