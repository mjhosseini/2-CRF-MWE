#I should have inheritance here, for now, just a quick implementation for SVM
from pyutil.ds import features
import numpy as np
from scipy.sparse import *
#from scipy import *
import sklearn
#from sklearn import svm
from decoding import legalTagBigramForLogistic
import pystruct
from scipy.optimize import fmin_l_bfgs_b
import scipy
import time
from libc.math cimport *
from cython.view cimport array as cvarray
import math
from numbers import Number
import sys
import dimsumeval
from streusle2dimsum import s2d



class SVM:

    def __init__(self,alpha):
        self._alpha = alpha


    def train(self,trainingData):
        print "inside svm"
        print "islegal:", legalTagBigramForLogistic("O-hi", "B-hi", 'NO_SINGLETON_B')
        Y = []
        YMap = {}
        nS = nW = 0
        #allIndices = set()

        _features = features.SequentialStringIndexer(cutoff=5)
        _labels = features.SequentialStringIndexer()


        for sent,o0FeatsEachToken in trainingData:
            nS += 1
            # if (nS==100):
            #     break


            for w,o0Feats in zip(sent,o0FeatsEachToken):
                _labels.add(w.gold)
                nW += 1
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]
                indices = [i for (i,v) in o0Feats.items()]
                for i in indices:
                    _features.add(str(i))
                    #allIndices.add(i)

        print "num samples:", nS
        #l = sorted(allIndices)
        _features.freeze()
        nF = len(_features.strings)
        print "num remained:", nF

        _labels.freeze()

        l2i = _labels.s2i
        self._labels = _labels


        self._features = _features
        i2s = _features.strings
        s2i = _features.s2i
        i2s_set = set(i2s)

        rows = []
        cols = []
        datas = []

        wI = 0
        nS = 0

        trainingData.reset()

        for sent,o0FeatsEachToken in trainingData:
            nS += 1
            # if (nS==100):
            #     break

            for w,o0Feats in zip(sent,o0FeatsEachToken):

                y = l2i[w.gold]
                Y.append(y)
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]

                for (i,v) in o0Feats.items():
                    if (str(i) in i2s_set):
                        featIndex = s2i[str(i)]
                        rows.append(wI);
                        cols.append(featIndex)
                        datas.append(1)
                wI += 1



        #Y = [l2i[y] for y in Y]
        rows = np.array(rows)
        cols = np.array(cols)
        datas = np.array(datas)
        nW= wI
        print "nW: ", nW
        print "rows:", rows
        print "cols:", cols
        print "datas: ", datas
        X = csr_matrix( (datas,(rows,cols)), shape=(nW,nF))
        #print X.todense()
        clf = sklearn.linear_model.LogisticRegression(C=1/self._alpha,solver='lbfgs',multi_class='multinomial')#multi_class='multinomial' another way: penalty='l1'

        print "training SVM"
        print "Y: ", Y
        clf.fit(X,Y)
        self._clf = clf
        print "SVM trained on data"

    def getBestTag(self, scores, sent, wI, nTokens, i2l):
        if wI==0:
            prevLabel = None
        else:
            prevLabel = sent[wI-1].prediction
        for i in range(len(i2l)):
            if legalTagBigramForLogistic(prevLabel, i2l[i],'NO_SINGLETON_B')==False:
                scores[i] -= 100000
        if wI==nTokens-1:
            for i in range(len(i2l)):
                if legalTagBigramForLogistic(i2l[i], None,'NO_SINGLETON_B')==False:
                    scores[i] -= 100000
        return i2l[np.argmax(scores)]



    def test(self,testData,outFilePath=None):
        gold_Ys = []

        nS = nW = 0

        if (outFilePath is not None):
            outFile = open(outFilePath, 'w')
        else:
            outFile = None

        _features = self._features
        nF = len(_features.strings)
        print "num remained:", nF



        i2s = _features.strings
        s2i = _features.s2i
        i2s_set = set(i2s)
        clf = self._clf

        i2l = self._labels.strings

        correctCount = 0
        all = 0

        for sent,o0FeatsEachToken in testData:
            nTokens = len(sent)
            wI = 0
            for w,o0Feats in zip(sent,o0FeatsEachToken):
                gold_Ys.append(w.gold)
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]
                indices = [i for (i,v) in o0Feats.items()]
                X_test = np.zeros(shape=(nF))

                for (i,v) in o0Feats.items():
                    if (str(i) in i2s_set):
                        featIndex = s2i[str(i)]
                        X_test[featIndex] = 1

                pr = clf.decision_function(X_test)[0,]
                print pr
                print pr.__class__


                pred = self.getBestTag(pr,sent,wI,nTokens,i2l)

                sent[wI] = sent[wI]._replace(prediction=pred)

                all += 1

                print "predict: ", pred, ", gold ", sent[wI]
                wI += 1

                if (pred==w.gold):
                    correctCount += 1
            sent.updatedPredictions_without_assert()
            if (outFile is not None):
                outFile.write(sent.tagsStr()+"\n\n")
        print "acc: ", (float)(correctCount)/all

cdef class CRF:

    cdef double _alpha, _alpha2, _inf, _cutoff, _numIter, _maxSentence
    cdef int _nF, _nLabels, _tt, _numIterMod, _iterCount
    cdef list _i2l, _i2s, _nextls, _prevls, _currentAlphas
    cdef _X_train, _Y_train
    cdef double _time0
    cdef int[:] _startls, _endls
    cdef double[:] _currentLambda, _lmbda
    cdef _features, _labels, _featNames
    cdef dict _comp_feats
    cdef _testData, _modOutFilePath, _testPath #This is just to print stats after a while...

    def __init__(self,alpha, alpha2,numIter, numIterMod, cutoff, testData, modOutFilePath, test_predict):
        self._alpha = alpha
        self._alpha2 = alpha2
        self._time0 = time.time()
        self._cutoff = cutoff
        self._numIter = numIter
        self._numIterMod = numIterMod
        self._maxSentence = -1
        self._iterCount = 0

        self._testData = testData
        self._modOutFilePath = modOutFilePath
        self._testPath = test_predict


        print "alpha: ", alpha
        print "alpha2: ", alpha2


    def train(self,trainingData):
        print "inside CRF"

        print "initialized!"



        (features, labels, X_train,Y_train,nF,nLabels,prevls,nextls,startls, endls, i2l,i2s,featNames, comp_feats) = self.extractXYs(trainingData)
        self._features = features
        self._labels = labels
        self._featNames = featNames
        self._comp_feats = comp_feats



        #crf = pystruct.models.ChainCRF(n_states = nLabels, n_features = nF)
        #from pystruct.models import ChainCRF
        #model = ChainCRF()
        #from pystruct.learners import FrankWolfeSSVM
        #crf = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
        self.fit(X_train, Y_train,nF,nLabels,prevls,nextls,startls, endls, i2l,i2s)
        #self.fitSGD(X_train, Y_train,nF,nLabels,prevls,nextls,startls, endls, i2l,i2s)

        print "crf trained on data"


    cdef fit(self,X_train,Y_train,nF,nLabels,prevls,nextls,int[:] startls, int[:] endls, i2l,i2s):
        nAllFeatures = nF*nLabels + nLabels*nLabels #0-order + 1-order
        print "nAllFeatures: ", nAllFeatures
        self._X_train = X_train
        self._Y_train = Y_train
        self._nF = nF
        self._nLabels = nLabels
        self._prevls = prevls
        self._nextls = nextls
        self._startls = startls
        self._endls = endls
        self._i2l = i2l
        self._i2s = i2s
        self._inf = 50
        self._currentAlphas = []
        self._currentLambda = None
        self._tt = 0


        cdef int i
        cdef double[:] lmbda0

        lmbda0 = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            lmbda0[i] = 0


        ret = fmin_l_bfgs_b(self.compute_ll_bfgs,lmbda0,fprime=self.compute_llDer_bfgs,maxiter=self._numIter,maxfun=np.int(self._numIter*1.2))
        print ret
        self._lmbda = ret[0]
        print "final lmbda: ",self._lmbda

        # print "1st orders:"
        # for l in range(nLabels):
        #     for i in range(nF):
        #         print self._i2l[l].encode('utf-8') +" "+ self._featNames[self._i2s[i]] + ": " + str(self._lmbda[i+l*nF])
        #
        # print "2nd orders:"
        # for l2 in range(nLabels):
        #     for l1 in range(nLabels):
        #         print self._i2l[l2].encode('utf-8')+" "+ self._i2l[l1].encode('utf-8') +": " + str(self._lmbda[nF*nLabels+ self.getYPairIdx(l1,l2,nLabels)])



    cdef fitSGD(self,X_train,Y_train,nF,nLabels,prevls,nextls,int[:] startls, int[:] endls, i2l,i2s):

        cdef int nSamples, nAllFeatures, i, nW
        cdef double[:] lmbda
        cdef double[:,:] alphas, betas
        cdef int[:] y_train

        nAllFeatures = nF*nLabels + nLabels*nLabels #0-order + 1-order
        print "nAllFeatures: ", nAllFeatures
        self._X_train = X_train
        self._Y_train = Y_train
        self._nF = nF
        self._nLabels = nLabels
        self._prevls = prevls
        self._nextls = nextls
        self._startls = startls
        self._endls = endls
        self._i2l = i2l
        self._i2s = i2s
        self._inf = 50
        self._currentAlphas = []
        self._currentLambda = None
        self._tt = 0




        lmbda = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            lmbda[i] = 0

        nSamples = len(X_train)
        nAllFeatures = nF*nLabels + nLabels*nLabels #0-order + 1-order

        ret = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            ret[i] = 0

        shouldComputeAlpha = True
        if shouldComputeAlpha:
            self._currentLambda = self.copyVec(self._currentLambda, lmbda)

        for SGDIter in range(60):
            for i in range(nSamples):
                x_train = X_train[i]
                y_train = Y_train[i]

                x_train_t = self.getIdxesSparse(x_train)
                if (shouldComputeAlpha):
                    #print "computing alhpas"
                    alphas = self.computeAlphas(x_train_t,lmbda,nF,nLabels,i)#TODO: you can save here!
                else:
                    alphas = self._currentAlphas[i]
                #print "computing betas"
                betas = self.computeBetas(x_train_t,lmbda,nF,nLabels)
                nW = x_train.shape[0]
                #print "compute derivation"
                self.computellDerOneSample(x_train_t,y_train,lmbda,alphas,betas,nW,nF,nLabels,lmbda,.1)

            ll = self.computell(self._X_train,self._Y_train,lmbda,self._nF,self._nLabels)
            print "ll: ",-ll



        #print "der penalty:", 2*self._alpha * lmbda
        # for i in range(nF*self._nLabels):
        #     ret[i] -= 2*self._alpha * lmbda[i]
        # for i in range(nF*self._nLabels,len(ret)):
        #     ret[i] -= 2*self._alpha2 * lmbda[i]
        self._lmbda = lmbda

        print "final lmbda: ",self._lmbda

    def compute_ll_bfgs(self,lmbda):
        print "computing ll"
        print "lmbda: ",lmbda
        ret = self.computell(self._X_train,self._Y_train,lmbda,self._nF,self._nLabels)
        print "ll: ",-ret
        return -ret


    def compute_llDer_bfgs(self,lmbda):
        print "computing ll Der"
        self._iterCount +=1
        ret = self.computellDer(self._X_train,self._Y_train,lmbda,self._nF,self._nLabels)
        for i in range(len(ret)):
            ret[i] = -ret[i]
        print "llDer: ",np.array(ret)


        #You may wanna comment out this completely
        if (self._iterCount%self._numIterMod==0):
            #Here, we're going to run the test for the current lmbda...
            self._lmbda = self._currentLambda

            dimSumOutPath = self._modOutFilePath[:self._modOutFilePath.rindex('.')]+ "-curCount=" + str(self._iterCount)
            outFilePath = dimSumOutPath+".tags"
            #Now, we have everything in outFilePath
            self.test(self._testData,outFilePath)

            #Convert to dimsum
            sys.argv=["",outFilePath,dimSumOutPath]
            s2d()


            #Now, run dimsumeval
            testPath = self._testPath[:self._testPath.rindex('.')]
            sys.argv = ["",testPath,dimSumOutPath]
            dimsumeval.evaluate()




        return np.array(ret)



    cdef logSumList(self,double[:] l,int length):

        cdef double ret
        cdef int i
        ret = 0
        minIdx = 0
        for i in range(length):
            if l[i]>l[minIdx]:
                minIdx = i

        #print "l: ",np.array(l)
        for i in range(length):

            ret = ret + exp(l[i] - l[minIdx])

        #x = np.sum(np.exp(x))

        ret = l[minIdx] + log(ret)
        if math.isnan(ret):
            print "l: ",np.array(l)
            print "lmbda: ", np.array(self._currentLambda)
        #print "logsum: ",ret


        return ret

    # def logSum(self,a,b):
    #
    #     if (a>b):
    #         ret = a + np.log(1+np.exp(b-a))
    #     else:
    #         ret = b + np.log(1+np.exp(a-b))
    #     return ret

    #This returns the log, because of over/under flows!
    cdef computeAlphas(self,x_train_t,double[:] lmbda,int nF,int nLabels,int sentIdx):


        cdef double[:] valList
        cdef double d0, d, fOrderLmbda0
        cdef int idx, i, curnW, l, l1, lp, l2
        cdef double[:,:] ret
        cdef list prevls
        cdef set startlsSet, endlsSet

        startlsSet = set(self._startls)
        endlsSet = set(self._endls)

        curnW = x_train_t[4]

        ret = cvarray(shape=(nLabels,curnW), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            for j in range(curnW):
                ret[l,j] = 0
        #We compute everything in log space

        prevls = self._prevls
        #Initialize
        time0 = self._time0

        for l in range(nLabels):
            l2 = 0
            l1 = l
            d = self.computeDotLambdaHLabels(x_train_t,l1,l2,0,lmbda,nF,nLabels)
            ret[l,0] = d
            if not l in startlsSet:
                ret[l,0] = -self._inf

        valList = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        for i in range(nLabels):
            valList[i] = 0

        for i in range(1,curnW):
            for l in range(nLabels):
                #val = 0
                l1 = l
                d0 = self.computeDotLambdaHLabels(x_train_t,l1,0,i,lmbda,nF,nLabels)
                #print "0: ", time.time() - time0

                fOrderLmbda0 = lmbda[nF*nLabels+self.getYPairIdx(l1,0,nLabels)]




                #print "1: ", time.time() - time0

                #print "t: ", time.time() - time0

                #if i==curnW-1 and not legalTagBigramForLogistic(self._i2l[l],None,True):
                if i==curnW-1 and not l in endlsSet:
                    ret[l,i] = -self._inf
                    continue


                #valList = []
                #print "2: ", time.time() - time0


                for idx,lp in enumerate(prevls[l]):

                    d = d0 - fOrderLmbda0 + lmbda[nF*nLabels+self.getYPairIdx(l1,lp,nLabels)]



                    #val = val + np.exp(ret[lp,i-1]+d)
                    valList[idx] = (ret[lp,i-1]+d)
                    #valList.append(ret[lp,i-1]+d)

                #print "val: ",val
                #print "3: ", time.time() - time0
                ret[l,i] = self.logSumList(valList,len(prevls[l]))
                #print "4: ", time.time() - time0

        #print ("res of compAlpha:", ret)
        if self._tt%200==0:
            print "5: ", time.time() - time0
        if sentIdx<len(self._currentAlphas):
            self._currentAlphas[sentIdx] = ret
        else:
            self._currentAlphas.append(ret)
        return ret

    #For now, I don't have log for this, but check if this might be necessary
    cdef predict(self,x_train_t,double[:] lmbda,int nF,int nLabels):

        cdef double d0, d, fOrderLmbda0, val
        cdef int idx, i, curnW, l, l1, lp, l2
        cdef double[:,:] ret
        cdef list prevls
        cdef set startlsSet, endlsSet
        cdef int[:,:] pointers
        cdef int[:] retPath

        startlsSet = set(self._startls)
        endlsSet = set(self._endls)
        curnW = x_train_t[4]

        ret = cvarray(shape=(nLabels,curnW), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            for j in range(curnW):
                ret[l,j] = 0

        pointers = cvarray(shape=(nLabels,curnW), itemsize=sizeof(int), format="i")
        for l in range(nLabels):
            for j in range(curnW):
                pointers[l,j] = 0

        retPath = cvarray(shape=(curnW,), itemsize=sizeof(int), format="i")
        for j in range(curnW):
            retPath[j] = 0

        prevls = self._prevls

        #Initialize
        for l in range(nLabels):
            l2 = 0
            l1 = l
            d = self.computeDotLambdaHLabels(x_train_t,l1,l2,0,lmbda,nF,nLabels)
            ret[l,0] = d
            if not l in startlsSet:
                ret[l,0] = -self._inf
            if curnW == 1 and not l in endlsSet:
                ret[l,0] = -self._inf



        for i in range(1,curnW):
            for l in range(nLabels):
                bestVal = -100000
                bestIdx = -1

                l1 = l
                d0 = self.computeDotLambdaHLabels(x_train_t,l1,0,i,lmbda,nF,nLabels)

                fOrderLmbda0 = lmbda[nF*nLabels+self.getYPairIdx(l1,0,nLabels)]


                if i==curnW-1 and not l in endlsSet:
                    ret[l,i] = -self._inf
                    continue

                for idx,lp in enumerate(prevls[l]):

                    d = d0 - fOrderLmbda0 + lmbda[nF*nLabels+self.getYPairIdx(l1,lp,nLabels)]

                    #d = d0 - fOrderLmbda0 + lmbda[nF*nLabels+l1*nLabels+lp]


                    #val = ret[lp,i-1]+d
                    val = ret[lp,i-1]+d
                    if val>bestVal:
                        bestVal = val
                        bestIdx = lp

                ret[l,i] = bestVal
                pointers[l,i] = bestIdx


        retPath[curnW-1] = np.argmax(ret[:,curnW-1])
        for i in range(curnW-1,0,-1):
            retPath[i-1] = pointers[retPath[i],i]
        return (retPath,ret)


    #This returns the log, because of over/under flows!
    cdef computePartitionFunction(self,alphas,nW,nLabels):
        cdef double ret
        cdef double[:] valList

        valList = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")

        ret = 0

        for l in range(nLabels):
            valList[l] = (alphas[l,nW-1])

        ret = self.logSumList(valList,nLabels)
        return ret



    #This returns the log, because of over/under flows!
    cdef computeBetas(self,x_train_t,double[:] lmbda,int nF,int nLabels):

        time0 = self._time0
        cdef double[:] valList
        cdef double d0, d, fOrderLmbda0
        cdef double[:] d0s, fOrderLmbda0s
        cdef int idx, i, curnW, l, l1, lp, l2
        cdef double[:,:] ret
        cdef list nextls
        cdef set endlsSet, startlsSet


        curnW = x_train_t[4]


        endlsSet = set(self._endls)
        startlsSet = set(self._startls)


        ret = cvarray(shape=(nLabels,curnW), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            for j in range(curnW):
                ret[l,j] = 0
        #We compute everything in log space

        nextls = self._nextls

        curnW = x_train_t[4]

        for l in range(nLabels):
            if not l in endlsSet:
                ret[l,curnW-1] = -self._inf

        d0s = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        fOrderLmbda0s = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")

        valList = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        for i in range(nLabels):
            valList[i] = 0

        for i in range(curnW-1,0,-1):
            for lp in range(nLabels):
                d0 = self.computeDotLambdaHLabels(x_train_t,lp,0,i,lmbda,nF,nLabels)
                fOrderLmbda0 = lmbda[nF*nLabels+self.getYPairIdx(lp,0,nLabels)]

                d0s[lp] = d0
                fOrderLmbda0s[lp] = fOrderLmbda0

            for l in range(nLabels):
                if i==0 and not l in startlsSet:
                    ret[l,i] = -inf
                    continue

                for idx, lp in enumerate(nextls[l]):
                    l1 = lp
                    l2 = l
                    d = d0s[l1] - fOrderLmbda0s[l1] + lmbda[nF*nLabels+self.getYPairIdx(l1,l2,nLabels)]
                    valList[idx] = ret[lp,i]+d

                ret[l,i-1] = self.logSumList(valList,len(nextls[l]))
        #print "res of comp betas: ", ret
        if self._tt%200==0:
            print "5b: ", time.time() - time0
        return ret


    cdef compareVec(self, double[:] l1, double[:] l2):
        cdef int i
        if (l1==None or l2==None):
            return False
        for i in range(len(l1)):
            if (l1[i]!=l2[i]):
                return False
        print "compare: True"
        return True

    cdef copyVec(self, double[:]l1, double[:] l2):
        if (l1==None or len(l1)!=len(l2)):
            l1 = cvarray(shape=(len(l2),), itemsize=sizeof(double), format="d")
        for i in range(len(l2)):
            l1[i] = l2[i]
        return l1

    cdef computell(self,X_train,Y_train,double[:] lmbda,int nF,int nLabels):
        nSamples = len(X_train)
        ret = 0

        shouldComputeAlpha = not self.compareVec(self._currentLambda,lmbda)

        if True:
            self._currentLambda = self.copyVec(self._currentLambda, lmbda)

        for i in range(nSamples):

            x_train = X_train[i]
            y_train = Y_train[i]

            (x_idxes,y_idxes,vals,wIdxes,curnW) = self.getIdxesSparse(x_train)
            x_train_t = (x_idxes,y_idxes,vals,wIdxes,curnW)

            #print "t1:",time.time() - self._time0
            if (shouldComputeAlpha):
                #print "computing alphas"
                alphas = self.computeAlphas(x_train_t,lmbda,nF,nLabels,i)
            else:
                alphas = self._currentAlphas[i]
            #print "t2:",time.time() - self._time0
            #betas = self.computeBetas(x_train,lmbda,nF,nLabels)

            #print "computing ll one sample"
            ret = ret + self.computellOneSample(x_train_t,y_train,lmbda,alphas,curnW,nF,nLabels)
            #print "t3:",time.time() - self._time0

        #print "alpha: ",self._alpha, "penalty: ",self._alpha * np.sum(lmbda*lmbda)
        ret -= (self._alpha * (np.linalg.norm(lmbda[:nF*nLabels])**2) + self._alpha2 * (np.linalg.norm(lmbda[nF*nLabels:])**2))
        return ret

    cdef computellDer(self,X_train,Y_train,double[:] lmbda,int nF,int nLabels):
        cdef int nSamples, nAllFeatures, i, nW
        cdef double[:] ret
        cdef double[:,:] alphas, betas
        cdef int[:] y_train


        nSamples = len(X_train)
        nAllFeatures = nF*nLabels + nLabels*nLabels #0-order + 1-order

        ret = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            ret[i] = 0

        shouldComputeAlpha = not self.compareVec(self._currentLambda,lmbda)
        if shouldComputeAlpha:
            self._currentLambda = self.copyVec(self._currentLambda, lmbda)

        for i in range(nSamples):
            x_train = X_train[i]
            y_train = Y_train[i]

            x_train_t = self.getIdxesSparse(x_train)
            if (shouldComputeAlpha):
                print "computing alhpas"
                alphas = self.computeAlphas(x_train_t,lmbda,nF,nLabels,i)#TODO: you can save here!
            else:
                alphas = self._currentAlphas[i]
            print "computing betas"
            betas = self.computeBetas(x_train_t,lmbda,nF,nLabels)
            nW = x_train.shape[0]
            print "compute derivation"
            self.computellDerOneSample(x_train_t,y_train,lmbda,alphas,betas,nW,nF,nLabels,ret)

        #print "der penalty:", 2*self._alpha * lmbda
        for i in range(nF*self._nLabels):
            ret[i] -= 2*self._alpha * lmbda[i]
        for i in range(nF*self._nLabels,len(ret)):
            ret[i] -= 2*self._alpha2 * lmbda[i]
        print "llDer: ",np.array(ret)
        return ret

    cdef computellOneSample(self,x_train_t,int[:] y_train,double[:] lmbda,double[:,:] alphas,int nW,int nF,int nLabels):

        ret = 0
        for j in range(nW):
            ret = ret + self.computeDotLambdaH(x_train_t,y_train,j,lmbda,nF,nLabels)
        ret = ret - self.computePartitionFunction(alphas,nW,nLabels)
        #print "llOne: ",ret
        return ret


    cdef getIdxesSparse(self,x):
        idxes = scipy.sparse.find(x)
        cdef int[:] x_idxes, y_idxes;
        cdef int[:] jidxes
        cdef double[:] vals
        cdef list wIdxes
        cdef int i,j,t
        x_idxes = idxes[0]
        y_idxes = idxes[1]
        vals = idxes[2]
        curnW = x.shape[0]
        wIdxes = []

        for j in range(curnW):
            wIdxes.append([])

        for i in range(len(x_idxes)):
            wIdxes[x_idxes[i]].append(i)

        for j in range(curnW):
            jidxes = np.array(wIdxes[j],dtype='int32')
            wIdxes[j] = jidxes

        # for j in range(curnW):
        #     jidxes = np.where(x_idxes==j)[0]
        #     #print "#feat: ",len(jidxes)
        #     wIdxes.append(jidxes)

        cdef list ret
        ret = [x_idxes,y_idxes,vals,wIdxes,curnW]
        return ret



    cdef computellDerOneSample(self,x_train_t,int[:] y_train,double[:] lmbda,double[:,:] alphas,double[:,:] betas,int nW,int nF,int nLabels,double[:] prevDer, double etha=1):
        #nAllFeatures = nF + nLabels*nLabels #0-order + 1-order
        #ret = np.zeros(nAllFeatures)
        self._tt += 1

        cdef int[:] x_idxes, y_idxes, jidxes
        cdef double[:] pl1s, vals
        cdef double[:,:] pl1l2s
        cdef double z, val, pl1, pl1l2, thisDot0, thisDot, fOrderLmbda0, thisAlpha, thisBeta
        cdef list wIdxes, prevls, i2l
        cdef int curnW, ii, l1, l2, wIdx, fNumber, j, ypairIdx
        cdef dict comp_feats

        comp_feats = self._comp_feats
        i2l = self._i2l

        (x_idxes,y_idxes,vals,wIdxes,curnW) = x_train_t

        prevls = self._prevls


        z = self.computePartitionFunction(alphas,nW,nLabels)
        #print "z: ", z

        #Second sum of derivation

        #print "up to here1"

        pl1s = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        pl1l2s = cvarray(shape=(nLabels,nLabels), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            pl1s[l] = 0
            for l2 in range(nLabels):
                pl1l2s[l,l2] = 0

        for j in range(nW):


            for l1 in range(nLabels):
                #pl1 = (1/z) * alphas[l1,j] * betas[l1,j]
                pl1 = -z + alphas[l1,j] + betas[l1,j]
                pl1s[l1] = exp(pl1)
            #print "j:", j, "pl1s: ",np.array(pl1s)

            if self._tt%100==0:
                print "sum pl1s: ",np.sum(pl1s)


            for l1 in range(nLabels):
                thisDot0 = self.computeDotLambdaHLabels(x_train_t,l1,0,j,lmbda,nF,nLabels)
                if (j>0):
                    fOrderLmbda0 = lmbda[nF*nLabels+self.getYPairIdx(l1,0,nLabels)]
                else:
                    fOrderLmbda0 = 0

                if (j!=0):
                    for l2 in prevls[l1]:
                        thisAlpha = 0
                        if j>0:
                            thisAlpha = alphas[l2,j-1]
                        #print "j: ",j," l1: ",l1
                        thisBeta = betas[l1,j]

                        if j==0:
                            thisDot = thisDot0 - fOrderLmbda0
                        else:
                            thisDot = thisDot0 - fOrderLmbda0 + lmbda[nF*nLabels+self.getYPairIdx(l1,l2,nLabels)]

                        #pl1l2 = (1/z) * thisAlpha * np.exp(thisDot) * thisBeta
                        pl1l2 = -z + thisAlpha + thisDot + thisBeta
                        pl1l2s[l1,l2] = exp(pl1l2)

                        # For the first word, we assume that we have a dummy label for the -1 index in the sentence! Other than that, prob will be zero.
                        if j==0 and l2!=0:
                            pl1l2s[l1,l2] = 0
                    #print "pl1l2s: ",pl1l2s

            if (j!=0 and self._tt%100==0):
                print "sum pl1l2s: ",np.sum(pl1l2s)



            #print "up to here2"

            jidxes = wIdxes[j]
            #np.where(x_idxes==j)[0]

            #0-order features
            for ii in jidxes:
                #wIdx = x_idxes[ii]

                #Conjoin with every possible label
                for l1 in range(nLabels):
                    val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])
                    pl1 = pl1s[l1]
                    fNumber = y_idxes[ii] + nF * l1
                    prevDer[fNumber] = prevDer[fNumber] - pl1*val*etha

            if (j==0):
                continue
            #Now, 1-order features:

            for l1 in range(nLabels):
                for l2 in prevls[l1]:
                    pl1l2 = pl1l2s[l1,l2]
                    fNumber = nF*nLabels + self.getYPairIdx(l1,l2,nLabels)
                    prevDer[fNumber] = prevDer[fNumber] - pl1l2*etha


        #First sum of derivation

        #0-order features: all nW * nFeatures for this sentence
        for ii in range(len(x_idxes)):
            wIdx = x_idxes[ii]
            l1 = y_train[wIdx]
            fNumber = y_idxes[ii] + nF * l1
            val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])

            #print "fnumber: ",fNumber
            prevDer[fNumber] += etha*val

        #1-order features

        for j in range(curnW):
            l1 = y_train[j]
            l2 = 0
            if j!=0:
                l2 = y_train[j-1]
            ypairIdx = self.getYPairIdx(l1,l2,nLabels)
            fNumber = nF*nLabels+ypairIdx



            if j!=0 and l2 in prevls[l1]:
                #print "fnumber: ",fNumber
                prevDer[fNumber] = prevDer[fNumber] + 1*etha


            #print "up to here3"

        # for fNumber in range(nAllFeatures):
        #     ret[fNumber] = self.computellDerOneSampleOneFeature(x_train,y_train,lmbda,alphas,betas,nW,nF,nLabels,fNumber)

        #return ret

    #This is not optimized
    # def computellDerOneSampleOneFeature(self,x_train,y_train,lmbda,alphas,betas,nW,nF,nLabels,fNumber):
    #     ret = 0
    #     for j in range(nW):
    #         if (fNumber<nF):#0-order features
    #             ret = ret + x_train[j,fNumber]
    #         else:#1-order features
    #             l1 = y_train[j]
    #             l2 = 0
    #             if j!=0:
    #                 l2 = y_train[j-1]
    #             ypairIdx = self.getYPairIdx(l1,l2,nLabels)
    #             if (ypairIdx==fNumber):
    #                 ret = ret + 1
    #
    #     for j in range(nW):
    #         for l1 in range(nLabels):
    #             for l2 in range(nLabels):
    #
    #                 fVal = 0
    #
    #                  if (fNumber<nF):#0-order features
    #                     fVal = x_train[j,fNumber]
    #                 else:#1-order features
    #                     l1 = y_train[j]
    #                     l2 = 0
    #                     if j!=0:
    #                         l2 = y_train[j-1]
    #                     ypairIdx = self.getYPairIdx(l1,l2,nLabels)
    #                     if (ypairIdx==fNumber):
    #                         fVal = 1
    #                 if fVal==0:
    #                     continue
    #
    #                 z = self.computePartitionFunction(alphas,nW,nLabels)#TODO: you can save time here by pre-computing
    #                 thisAlpha = 1
    #                 if j>0:
    #                     thisAlpha = alphas[l2,j-1]
    #                 #print "j: ",j," l1: ",l1
    #                 thisBeta = betas[l1,j]
    #                 thisDot = self.computeDotLambdaHLabels(x_train,l1,l2,j,lmbda,nF,nLabels)
    #                 pl1l2 = (1/z) * thisAlpha * thisDot * thisBeta
    #
    #                 ret = ret - pl1l2 * fVal
    #
    #     return ret


    #The input is X_train[i] for some i: nW*nF for a sentence
    cdef computeDotLambdaH(self,x_train_t,int[:] y_train, int j, double[:] lmbda, int nF, int nLabels):
        cdef int l1,l2
        l1 = y_train[j]
        l2 = 0
        if j!=0:
            l2 = y_train[j-1]

        return self.computeDotLambdaHLabels(x_train_t,l1,l2,j,lmbda,nF,nLabels)

    #The input is X_train[i] for some i: nW*nF for a sentence
    #l1 is the current label, l2 is the previous one
    cdef computeDotLambdaHLabels(self,x_train_t, int l1, int l2, int j,double[:] lmbda, int nF, int nLabels):
        #print "x_train: ",x_train.todense()

        cdef int[:] x_idxes, y_idxes
        cdef int[:] jidxes
        cdef int curnW
        cdef double[:] vals
        cdef list wIdxes, i2l
        cdef double ret, val
        cdef dict comp_feats

        comp_feats = self._comp_feats
        i2l = self._i2l

        (x_idxes,y_idxes,vals,wIdxes,curnW) = x_train_t


        jidxes = wIdxes[j]
        #np.where(x_idxes==j)[0]



        #ll = [lmbda[y_idxes[ii] + nF * l1] * vals[ii] for ii in jidxes]
        ret = 0
        for ii in jidxes:

            val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])

            ret += lmbda[y_idxes[ii] + nF * l1] * val

        #ret = ret + np.sum(ll)

        # for ii in jidxes:
        #     fNumber = y_idxes[ii] + nF * l1
        #     ret = ret + lmbda[fNumber] * vals[ii]



        if (j>0):
            ret = ret + lmbda[nF*nLabels+self.getYPairIdx(l1,l2,nLabels)]
        return ret

    cdef getYPairIdx(self,int l1,int l2,int nLabels):
        return l1*nLabels+l2


    cdef getLegalNextLabels(self,i2ls):
        cdef list nextls, prevls
        cdef int[:] startls, endls
        cdef int l, i

        print "legals:"

        nextls =[[] for i in range(len(i2ls))]
        prevls = [[] for i in range(len(i2ls))]
        for l in range(len(i2ls)):
            for l2 in range(len(i2ls)):
                if legalTagBigramForLogistic(i2ls[l],i2ls[l2],'NO_SINGLETON_B'):
                    print i2ls[l].encode('utf-8')," ",i2ls[l2].encode('utf-8')
                    nextls[l].append(l2)
                if legalTagBigramForLogistic(i2ls[l2],i2ls[l],'NO_SINGLETON_B'):
                    prevls[l].append(l2)


        ends = []
        starts = []
        for l in range(len(i2ls)):
            if legalTagBigramForLogistic(i2ls[l],None,'NO_SINGLETON_B'):
                ends.append(l)
                print "end: ", i2ls[l].encode('utf-8')
            if legalTagBigramForLogistic(None, i2ls[l],'NO_SINGLETON_B'):
                starts.append(l)
        #print "start:", starts
        #print "end:", ends

        endls = np.array(ends, dtype='int32')
        startls = np.array(starts, dtype='int32')

        # for l in range(len(i2ls)):
        #     #ls = np.array(nextls[l],dtype='int32')
        #     ls = cvarray(shape=(len(nextls[l]),), itemsize=sizeof(int), format="i")
        #     for k in range(len(nextls[l])):
        #         ls[k] = nextls[l][k]
        #     nextls[l] = ls
        #
        #     #ls = np.array(prevls[l],dtype='int32')
        #     ls = cvarray(shape=(len(prevls[l]),), itemsize=sizeof(int), format="i")
        #     for k in range(len(prevls[l])):
        #         ls[k] = prevls[l][k]
        #     prevls[l] = ls

        print nextls
        print prevls
        lens = [len(x) for x in nextls]
        print "all lens:", lens
        print "all legal lens:",np.sum(lens)
        return (prevls,nextls, startls, endls)

    cdef extractXYs(self, trainingData):

        cdef int[:] Ycur
        cdef dict comp_feats
        cdef double[:] valsArr

        comp_feats = {}
        Y = []
        YMap = {}
        nS = nW = 0
        #allIndices = set()

        _features = features.SequentialStringIndexer(cutoff=self._cutoff)
        _labels = features.SequentialStringIndexer()
        _featureNames = {}

        print "feat names: ", trainingData._featureIndexes

        for i,pname in trainingData._featureIndexes.items():
            print i," ",pname


        for sent,o0FeatsEachToken in trainingData:
            nS += 1
            if (self._maxSentence>0 and nS==self._maxSentence):
                break

            for w,o0Feats in zip(sent,o0FeatsEachToken):
                nW += 1
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]
                #o0Feats is a pyutil.ds.IndexedFeatureMap


                _labels.add(w.gold)

                indices = [i for (i,v) in o0Feats.items()]
                for i in indices:
                    _features.add(str(i))
                    _featureNames[str(i)] = str(o0Feats._set._indexer[i])
                    #print i,":",o0Feats._set._indexer[i]," ",o0Feats._map.get(i, o0Feats._default)
                    #allIndices.add(i)


        _labels.freeze()

        l2i = _labels.s2i
        #self._labels = _labels
        i2l = _labels.strings



        #l = sorted(allIndices)
        _features.freeze()
        nF = len(_features.strings)
        print "num remained:", nF


        #self._features = _features
        i2s = _features.strings
        s2i = _features.s2i
        i2s_set = set(i2s)

        wI = 0
        nS = 0

        trainingData.reset()

        X_train = []
        Y_train = []
        X_train_sparse = []

        for sent,o0FeatsEachToken in trainingData:
            nS += 1
            if (self._maxSentence>0 and nS==self._maxSentence):
                 break

            #Xcur = np.zeros(shape=(len(sent),nF))
            rowsCur = []
            colsCur = []
            valsCur = []
            #Ycur = np.zeros(shape=(len(sent)),dtype='int32')
            Ycur = cvarray(shape=(len(sent),), itemsize=sizeof(int), format="i")

            wIdxCur = 0

            for w,o0Feats in zip(sent,o0FeatsEachToken):
                y = l2i[w.gold]
                Y.append(y)
                Ycur[wIdxCur] = y
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]
                indices = [i for (i,v) in o0Feats.items()]

                for (i,v) in o0Feats.items():
                    #print ("i:",i,"v:",v)
                    if (str(i) in i2s_set):
                        featIndex = s2i[str(i)]
                        # Xcur[wIdxCur,featIndex] = 1
                        rowsCur.append(wIdxCur);
                        colsCur.append(featIndex)

                        if not isinstance(v,Number):
                            if featIndex not in comp_feats:
                                print "adding ", featIndex, " ", i
                                comp_feats[featIndex] = v

                        #-1 means that we should compute a function. The function is accessible from comp_feats
                        val = -1 if not isinstance(v,Number) else v
                        valsCur.append(val)


                wI += 1
                wIdxCur = wIdxCur + 1
            #X_train.append(Xcur)

            valsArr = cvarray(shape=(len(valsCur),), itemsize=sizeof(double), format="d")
            for ii in range(len(valsCur)):
                valsArr[ii] = valsCur[ii]

            X_cur_sparse = csr_matrix( (valsArr,(rowsCur,colsCur)), shape=(wIdxCur,nF))
            X_train_sparse.append((X_cur_sparse))
            Y_train.append(Ycur)



        nW= wI
        print "nW: ", nW

        #X = csr_matrix( (datas,(rows,cols)), shape=(nW,nF))
        #print X.todense()
        nLabels = len(l2i)




        print X_train
        print Y_train

        cdef int[:] starts, ends

        X_train_sparse = np.array(X_train_sparse)
        #X_train = array(X_train)
        Y_train = np.array(Y_train)
        #print X_train_sparse[0].todense()
        print "nlabels: ", len(i2l)
        (prevls,nextls, starts, ends) = self.getLegalNextLabels(i2l)
        return (_features, _labels, X_train_sparse,Y_train,nF,nLabels,prevls,nextls,starts, ends,i2l,i2s, _featureNames, comp_feats)


    # cdef getBestTag(self, scores, sent, wI, nTokens, i2l):
    #     if wI==0:
    #         prevLabel = None
    #     else:
    #         prevLabel = sent[wI-1].prediction
    #     for i in range(len(i2l)):
    #         if legalTagBigramForLogistic(prevLabel, i2l[i],'NO_SINGLETON_B')==False:
    #             scores[i] -= 100000
    #     if wI==nTokens-1:
    #         for i in range(len(i2l)):
    #             if legalTagBigramForLogistic(i2l[i], None,'NO_SINGLETON_B')==False:
    #                 scores[i] -= 100000
    #     return i2l[np.argmax(scores)]

    def test(self,testData,outFilePath=None):



        nS = nW = 0

        if (outFilePath is not None):
            outFile = open(outFilePath, 'w')
        else:
            outFile = None

        lmbda = self._lmbda#: This is all we need to the classification

        _features = self._features
        nF = len(_features.strings)
        print "num features:", nF

        i2s = _features.strings
        s2i = _features.s2i
        i2s_set = set(i2s)


        i2l = self._labels.strings
        l2i = self._labels.s2i
        nLabels = len(i2l)

        correctCount = 0
        all = 0

        for sent,o0FeatsEachToken in testData:

            rowsCur = []
            colsCur = []
            #Ycur = np.zeros(shape=(len(sent)),dtype='int32')
            Ycur = cvarray(shape=(len(sent),), itemsize=sizeof(int), format="i")

            wIdxCur = 0

            nTokens = len(sent)

            for w,o0Feats in zip(sent,o0FeatsEachToken):

                #This is just for debugging with fewer sentences
                y = -1
                if w.gold in i2l:
                    y = l2i[w.gold]
                Ycur[wIdxCur] = y
                #print w, ":"
                #print [(i,v) for (i,v) in o0Feats.items()]
                indices = [i for (i,v) in o0Feats.items()]


                for (i,v) in o0Feats.items():
                    if (str(i) in i2s_set):
                        featIndex = s2i[str(i)]
                        rowsCur.append(wIdxCur)
                        colsCur.append(featIndex)

                wIdxCur += 1



                all += 1

            x_train = csr_matrix( (np.ones(len(rowsCur)),(rowsCur,colsCur)), shape=(wIdxCur,nF))

            x_train_t = self.getIdxesSparse(x_train)


            #change
            (preds,probs) = self.predict(x_train_t,self._lmbda,nF,nLabels)
            nTokens = len(sent)

            for j in range(wIdxCur):
                #Change
                # w = sent[j]
                # pred = self.getBestTag(probs[:,j],sent,j,nTokens,i2l)
                # sent[j] = sent[j]._replace(prediction=pred)
                # if (pred==w.gold):
                #     correctCount += 1

                pred = i2l[preds[j]]

                sent[j] = sent[j]._replace(prediction=pred)
                print "predict: ", pred.encode('utf-8'), ", gold ", sent[j].__str__().encode('utf-8')

                if (preds[j]==Ycur[j]):
                    correctCount += 1







            sent.updatedPredictions_without_assert()
            if (outFile is not None):
                outFile.write(sent.tagsStr()+"\n\n")
        print "acc: ", (float)(correctCount)/all
