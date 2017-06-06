import codecs
import dimsumeval
import numpy as np
import sys


#streusle = codecs.open(source+'/split/streusle.upos_all.tags','r','utf-8')

args = sys.argv[1:]

#source = '/Users/hosseini/Documents/python/pysupersensetagger-2.0/dimsum-dataAnalyzer-1.5/'
#addr1 = "split/comp/dimsum16.test.pred1"
#addr2 = "split/comp/dimsum16.test.pred2"
#addrGold = "dimsum16.test.gold"

source = args[0]
addr1 = args[1]
addr2 = args[2]
addrGold = args[3]



N=1000

def getF1sRandom(lines1,lines2):
    addr1tmp = "f1"+addr1.replace("/","")+addr2.replace("/","")
    addr2tmp = "f2"+addr1.replace("/","")+addr2.replace("/","")
    file1p = codecs.open(source+'split/temp/'+addr1tmp,'w','utf-8')
    file2p = codecs.open(source+'split/temp/'+addr2tmp,'w','utf-8')

    lBuffer1 = []
    lBuffer2 = []
    for (i,l) in enumerate(lines1):
        lBuffer1.append(l)
        lBuffer2.append(lines2[i])
        if (l.isspace()):
            if np.random.rand()<.5:
                file1p.writelines(lBuffer1)
                file2p.writelines(lBuffer2)
            else:
                file2p.writelines(lBuffer1)
                file1p.writelines(lBuffer2)
            lBuffer1 = []
            lBuffer2 = []

    args = [source+addrGold,source+'split/temp/'+addr1tmp]
    f1sp_M1 = dimsumeval.evaluate(args)

    args = [source+addrGold,source+'split/temp/'+addr2tmp]
    f1sp_M2 = dimsumeval.evaluate(args)

    return  (f1sp_M1,f1sp_M2)

file1 = codecs.open(source+addr1,'r','utf-8')
file2 = codecs.open(source+addr2,'r','utf-8')

fileGold = codecs.open(source+addrGold,'r','utf-8')

args = [source+addrGold,source+addr1]
f1s_M1 = dimsumeval.evaluate(args)

print "f1s_M1:" , f1s_M1

args = [source+addrGold,source+addr2]
f1s_M2 = dimsumeval.evaluate(args)

print "f1s_M2:" , f1s_M2

lines1 = file1.readlines()
lines2 = file2.readlines()


diffF1s = f1s_M1-f1s_M2

ncs = np.zeros(3)

for i in range(N):

    (f1sp_M1,f1sp_M2) = getF1sRandom(lines1,lines2)

    print "f1sp_M1:" , f1sp_M1
    print "f1sp_M2:" , f1sp_M2

    for j in range(3):
        diff1 = f1sp_M1[j]-f1sp_M2[j]
        diff2 = f1s_M1[j]-f1s_M2[j]
        if np.abs(diff1)>=np.abs(diff2) and diff1*diff2>=0:
            ncs[j] +=1
print "ncs: ",ncs

if (ncs[0]+1.0)/(N+1)<.05:
    print "MWE Significant"
else:
    print "MWE not Significant"

if (ncs[1]+1.0)/(N+1)<.05:
    print "SST Significant"
else:
    print "SST not Significant"

if (ncs[2]+1.0)/(N+1)<.05:
    print "Combined Significant"
else:
    print "Combined not Significant"






