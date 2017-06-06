#It reads streusle data, and splits it into train and test, based on train.sentids and test.sentids!
import codecs
source = '../streusle-2.0'
#streusle = codecs.open(source+'/split/streusle.upos_all.tags','r','utf-8')
streusle = codecs.open(source+'/split/streusle_all.tags','r','utf-8')
trainSentIdsFile = codecs.open(source+'/split/train.sentids','r','utf-8')
testSentIdsFile = codecs.open(source+'/split/test.sentids','r','utf-8')
trainFile = codecs.open(source+'/split/streusle.train_all.tags','w','utf-8')
testFile = codecs.open(source+'/split/streusle.test_all.tags','w','utf-8')
testFileDimSum = codecs.open(source+'/split/test_all.gold','w','utf-8')
testFilePreTrained = codecs.open(source+'/split/test_all_for_pre.test','w','utf-8')

trainSentIds = trainSentIdsFile.readlines()
testSentIds = testSentIdsFile.readlines()

lBuffer = []
sentId = None;

def getLabel(tag,label):
    # if (tag=="I" or tag =="i"):
    #     return ""
    if (tag is not "O" and tag is not "o" and tag is not "B" and tag is not"b"):
        return ""
    if label=="":
        return label
    if label.lower()==label:
        ret = "v."+label.lower()
    else:
        ret = "n."+label.lower()
    return ret

def getTag(tag):
    s = tag.split("-")
    return s[0]

#To note: For streusle itself, don't use this function
#This is for generating dimsum output: test.gold.
def streusleToDimSUM(line):
    if 1==1:
        return line
    line = line.encode('utf-8')
    strs = line.split("\t")
    for i in range(len(strs)):
        strs[i] = strs[i].encode('utf-8')
    tag = strs[4] = getTag(strs[4])
    strs[7] = getLabel(tag, strs[7])
    strs[6] = ""
    ret = ""
    #print "strs: ", strs
    for s in strs:
        ret += s.encode('utf-8')+"\t"
    ret = ret.strip()
    return ret

def streusleToPre(line):


    strs = line.split("\t")

    ret = ""
    #print "strs: ", strs
    ret += strs[1] +"\t"+ strs[3]
    return ret

for l in streusle:
    print (l)
    if (l.isspace()):
        print ("empty string")
        if (sentId in trainSentIds):
            trainFile.writelines(lBuffer)
            trainFile.write('\n')
        elif sentId in testSentIds:
            testFile.writelines(lBuffer)
            for ll in lBuffer:
                testFileDimSum.write(streusleToDimSUM(ll)+"\n")
                testFilePreTrained.write(streusleToPre(ll)+"\n")
            testFileDimSum.write("\n")
            testFilePreTrained.write("\n")
            testFile.write('\n')
        else:
            raise Exception('sentId not in train or test!')
        lBuffer = []
    else:
        print ('add to lBuffer')
        lBuffer.append(l)
        splits = l.split('\t')
        sentId = splits[len(splits)-1]

