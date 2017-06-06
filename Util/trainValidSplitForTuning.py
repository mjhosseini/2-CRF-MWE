#It reads streusle data, and splits it into train and test, based on train.sentids and test.sentids!
import codecs
#source = '../streusle-2.0'
source = '../dimsum-dataAnalyzer-1.5'
#streusle = codecs.open(source+'/split/streusle.upos_all.tags','r','utf-8')

#streusle
# trainFile = codecs.open(source+'/split/streusle.train_all.tags','r','utf-8')
# trainFileT = codecs.open(source+'/split/streusle.train_allT.tags','w','utf-8')#The portion to be used for training
# trainFileV = codecs.open(source+'/split/streusle.train_allV.tags','w','utf-8')#The portion to be used for training

#dimsum
trainFile = codecs.open(source+'/dimsum16.train','r','utf-8')
#trainFileT = codecs.open(source+'/split/dimsum16.train_allT.tags','w','utf-8')#The portion to be used for training
#trainFileV = codecs.open(source+'/split/dimsum16.train_allV.tags','w','utf-8')#The portion to be used for training

numValid = 0
numAll = 0
lBuffer = []

import random

for l in trainFile:
    print (l)
    if (l.isspace()):
        print ("empty string")
        numAll +=1
        if (random.random()>.3):
            numValid += 1
            # trainFileV.writelines(lBuffer)
            # trainFileV.write('\n')
        #else:
            # trainFileT.writelines(lBuffer)
            # trainFileT.write('\n')

        lBuffer = []
    else:
        print ('add to lBuffer')
        lBuffer.append(l)


        #These were for streusle!
        # splits = l.split('\t')
        # sentId = splits[len(splits)-1]
        # sentId = sentId[:sentId.rindex('.')]
        # sentId = sentId[sentId.rindex('.')+1:]
        #
        # for firstIdx in range(len(sentId)):
        #     if sentId[firstIdx] != '0':
        #         break
        # print "firstIdx:", firstIdx, " ", sentId
        # sentId = int(sentId[firstIdx:])
        # print "sentId: ",sentId

print numValid
print numAll