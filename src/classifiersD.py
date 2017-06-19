'''
Created on Dec 20, 2015

@author: Javad Hosseini (mjhosseini)
'''

from pyutil.ds import features
import numpy as np
from scipy.sparse import *
from decoding import legalTagBigramForLogistic
from scipy.optimize import fmin_l_bfgs_b
import scipy
import time
from libc.math cimport *
from cython.view cimport array as cvarray

from pyutil.memoize import memoize
from numbers import Number
import sys
import dimsumeval
from streusle2dimsum import s2d

@memoize
def getYSFromL(l,labels,labels1,labels2):

    s = labels.strings[l].split("-")
    l1 = labels1.s2i[s[0]]#BIO
    if len(s)>1:
        l2 = labels2.s2i[s[1]]
    else:
        l2 = labels2.s2i['0']
    return (l1,l2)


cdef class CRFD:

    cdef double _alpha, _alpha2, _inf
    cdef int _nF, _nLabels, _nLabels1, _nLabels2, _tt, _cutoff, _numIter, _maxSentence, _off1, _off2, _off3, _off4, _numIterMod, _iterCount
    cdef list _i2l, _i2s, _nextls, _prevls, _currentAlphas
    cdef _X_train, _Y_train, _Y_train1,_Y_train2
    cdef double _time0
    cdef int[:] _startls, _endls
    cdef int[:,:] _l2ys
    cdef double[:] _currentLambda, _lmbda
    cdef _features, _labels, _labels1, _labels2, _featNames
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
        print "alpha: ", alpha
        print "alpha2: ", alpha2

        self._testData = testData
        self._modOutFilePath = modOutFilePath
        self._testPath = test_predict


    def train(self,trainingData):

        print "CRF initialized!"



        (_features, _labels, _labels1,_labels2, X_train,Y_train,Y_train1,Y_train2,nF,nLabels,prevls,nextls,startls, endls,i2l,i2s,l2ys,featNames, comp_feats) = self.extractXYs(trainingData)
        self._features = _features
        self._labels = _labels
        self._labels1 = _labels1
        self._labels2 = _labels2
        self._featNames = featNames
        self._comp_feats = comp_feats

        nLabels1 = len(_labels1.strings)
        nLabels2 = len(_labels2.strings)


        self.fit(X_train, Y_train,Y_train1,Y_train2,nF,nLabels,nLabels1,nLabels2,prevls,nextls,startls, endls, i2l,i2s,l2ys)

        print "CRF trained"


    cdef fit(self,X_train,Y_train,Y_train1,Y_train2,int nF,int nLabels,int nLabels1,int nLabels2,list prevls,list nextls,int[:] startls, int[:] endls, i2l,i2s,int[:,:] l2ys):
        cdef int nAllFeatures
        nAllFeatures = nF*(nLabels1+nLabels2) + nLabels1**2 + nLabels2**2 + nLabels1*nLabels2 #0-order + 1-orders
        print "nAllFeatures: ", nAllFeatures
        self._X_train = X_train
        self._Y_train = Y_train
        self._Y_train1 = Y_train1
        self._Y_train2 = Y_train2
        self._nF = nF
        self._nLabels = nLabels
        self._nLabels1 = nLabels1
        self._nLabels2 = nLabels2
        self._off1 = nF*nLabels1
        self._off2 = nF*(nLabels1+nLabels2)
        self._off3 = self._off2 + nLabels1**2
        self._off4 = self._off3 + nLabels2**2


        self._prevls = prevls
        self._nextls = nextls
        self._startls = startls
        self._endls = endls
        self._i2l = i2l
        self._i2s = i2s
        self._l2ys = l2ys
        self._inf = 50
        self._currentAlphas = []
        self._currentLambda = None
        self._tt = 0


        cdef double[:] lmbda0

        lmbda0 = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            lmbda0[i] = 0


        ret = fmin_l_bfgs_b(self.compute_ll_bfgs,lmbda0,fprime=self.compute_llDer_bfgs,maxiter=self._numIter,maxfun=np.int(self._numIter*1.2))

        self._lmbda = ret[0]



    def compute_ll_bfgs(self,lmbda):
        print "computing likelihood"
        ret = self.computell(self._X_train,self._Y_train,self._Y_train1,self._Y_train2,lmbda,self._nF,self._nLabels)
        print "likelihood: ",-ret
        return -ret


    def compute_llDer_bfgs(self,lmbda):
        print "computing derivative"
        self._iterCount +=1
        ret = self.computellDer(self._X_train,self._Y_train,self._Y_train1,self._Y_train2,lmbda,self._nF,self._nLabels)
        for i in range(len(ret)):
            ret[i] = -ret[i]
        print "derivative: ",np.array(ret)


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

        for i in range(length):

            ret += exp(l[i] - l[minIdx])

        ret = l[minIdx] + log(ret)

        return ret


    cdef computellDerOneSample(self,x_train_t,int[:] y_train,int[:] y_train1,int[:] y_train2,double[:] lmbda,double[:,:] alphas,double[:,:] betas,int nW,int nF,int nLabels,double[:] prevDer):
        self._tt += 1

        cdef int[:] x_idxes, y_idxes, jidxes
        cdef double[:] pl1s, vals
        cdef double[:,:] pl1l2s
        cdef double z, val, pl1, pl1l2, thisDot0, thisDot, fOrderLmbda0, thisAlpha, thisBeta
        cdef list wIdxes, prevls, i2l
        cdef int curnW, ii, l1, l2, wIdx, fNumber, j, ypairIdx,s1, y1, s2, y2, nLabels12, nLabels1, nLabels2
        cdef dict comp_feats

        comp_feats = self._comp_feats
        i2l = self._i2l

        nLabels1 = self._nLabels1
        nLabels2 = self._nLabels2
        nLabels12 = self._nLabels1 + self._nLabels2

        (x_idxes,y_idxes,vals,wIdxes,curnW) = x_train_t

        prevls = self._prevls


        z = self.computePartitionFunction(alphas,nW,nLabels)

        #First sum of derivation

        #0-order features: all nW * nFeatures for this sentence
        for ii in range(len(x_idxes)):
            wIdx = x_idxes[ii]
            l1 = y_train[wIdx]

            y1 = y_train1[wIdx]
            s1 = y_train2[wIdx]
            val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])

            #BIO features
            prevDer[y_idxes[ii] + nF * y1] += val
            #sst features
            prevDer[y_idxes[ii] + self._off1 + nF * s1] += val


        #1-order features

        for j in range(curnW):
            l1 = y_train[j]
            y1 = y_train1[j]
            s1 = y_train2[j]

            l2 = 0
            s2 = 0
            y2 = 0

            if j!=0:
                l2 = y_train[j-1]
                y2 = y_train1[j-1]
                s2 = y_train2[j-1]

            if (j>0 and l2 in prevls[l1]):
                prevDer[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] += 1
                prevDer[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)] += 1

            prevDer[self._off4 + self.getYPairIdx(s1,y1,self._nLabels1)] += 1


        #Second sum of derivation

        pl1s = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        pl1l2s = cvarray(shape=(nLabels,nLabels), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            pl1s[l] = 0
            for l2 in range(nLabels):
                pl1l2s[l,l2] = 0

        for j in range(nW):

            for l1 in range(nLabels):
                pl1 = -z + alphas[l1,j] + betas[l1,j]
                pl1s[l1] = exp(pl1)


            for l1 in range(nLabels):
                (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])



                thisDot0 = self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,0,0,0,j,lmbda,nF,nLabels)


                fOrderLmbda0 = lmbda[self._off2+self.getYPairIdx(y1,0,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,0,self._nLabels2)]

                if (j!=0):
                    for l2 in prevls[l1]:
                        thisAlpha = 0
                        if j>0:
                            thisAlpha = alphas[l2,j-1]

                        thisBeta = betas[l1,j]

                        (y2,s2) = (self._l2ys[l2,0],self._l2ys[l2,1])


                        thisDot = thisDot0 - fOrderLmbda0 + lmbda[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)]

                        pl1l2 = -z + thisAlpha + thisDot + thisBeta
                        pl1l2s[l1,l2] = exp(pl1l2)


                        # For the first word, we assume that we have a dummy label for the -1 index in the sentence! Other than that, prob will be zero.
                        if j==0 and l2!=0:
                            pl1l2s[l1,l2] = 0

            jidxes = wIdxes[j]


            #0-order features
            for ii in jidxes:

                #Conjoin with every possible label
                for l1 in range(nLabels):
                    pl1 = pl1s[l1]

                    (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])
                    val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])

                    #BIO features
                    prevDer[y_idxes[ii] + nF * y1] -= pl1*val
                    #sst features
                    prevDer[y_idxes[ii] + self._off1 + nF * s1] -= pl1*val

            #Now, 1-order features:

            for l1 in range(nLabels):
                pl1 = pl1s[l1]
                (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])
                prevDer[self._off4 + self.getYPairIdx(s1,y1,self._nLabels1)] -= pl1

            if (j==0):
                continue

            for l1 in range(nLabels):
                (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])
                for l2 in prevls[l1]:
                    (y2,s2) = (self._l2ys[l2,0],self._l2ys[l2,1])
                    pl1l2 = pl1l2s[l1,l2]

                    prevDer[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] -= pl1l2
                    prevDer[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)] -= pl1l2

    #This returns the log, because of over/under flows!
    cdef computeAlphas(self,x_train_t,double[:] lmbda,int nF,int nLabels,int sentIdx):


        cdef double[:] valList
        cdef double d0, d, fOrderLmbda0
        cdef int idx, i, curnW, l, l1, lp, l2, s1, y1, s2, y2, nLabels12
        cdef double[:,:] ret
        cdef list prevls
        cdef set startlsSet, endlsSet

        startlsSet = set(self._startls)
        endlsSet = set(self._endls)

        curnW = x_train_t[4]

        nLabels12 = self._nLabels1 + self._nLabels2

        ret = cvarray(shape=(nLabels,curnW), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            for j in range(curnW):
                ret[l,j] = 0

        #We compute everything in log space

        prevls = self._prevls
        #Initialize
        time0 = self._time0

        for l in range(nLabels):
            l1 = l
            (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])


            d = self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,0,0,0,0,lmbda,nF,nLabels)
            ret[l,0] = d
            if not l in startlsSet:
                ret[l,0] = -self._inf

        valList = cvarray(shape=(nLabels,), itemsize=sizeof(double), format="d")
        for i in range(nLabels):
            valList[i] = 0

        for i in range(1,curnW):
            for l in range(nLabels):
                l1 = l
                (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])



                d0 = self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,0,0,0,i,lmbda,nF,nLabels)

                fOrderLmbda0 = lmbda[self._off2+self.getYPairIdx(y1,0,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,0,self._nLabels2)]


                if i==curnW-1 and not l in endlsSet:
                    ret[l,i] = -self._inf
                    continue

                for idx,lp in enumerate(prevls[l]):

                    (y2,s2) = (self._l2ys[lp,0],self._l2ys[lp,1])

                    d = d0 - fOrderLmbda0 + \
                    lmbda[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)]

                    valList[idx] = (ret[lp,i-1]+d)

                ret[l,i] = self.logSumList(valList,len(prevls[l]))


        if sentIdx<len(self._currentAlphas):
            self._currentAlphas[sentIdx] = ret
        else:
            self._currentAlphas.append(ret)
        return ret


    #This returns the log, because of over/under flows!
    cdef computeBetas(self,x_train_t,double[:] lmbda,int nF,int nLabels):

        time0 = self._time0
        cdef double[:] valList
        cdef double d0, d, fOrderLmbda0
        cdef double[:] d0s, fOrderLmbda0s
        cdef int idx, i, curnW, l, l1, lp, l2, s1, y1, s2, y2, nLabels12

        cdef double[:,:] ret
        cdef list nextls
        cdef set endlsSet, startlsSet


        curnW = x_train_t[4]


        endlsSet = set(self._endls)
        startlsSet = set(self._startls)

        nLabels12 = self._nLabels1 + self._nLabels2


        ret = cvarray(shape=(nLabels,curnW), itemsize=sizeof(double), format="d")
        for l in range(nLabels):
            for j in range(curnW):
                ret[l,j] = 0
        #We compute everything in log space

        nextls = self._nextls



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


                (y1,s1) = (self._l2ys[lp,0],self._l2ys[lp,1])

                d0 = self.computeDotLambdaHLabels(x_train_t,lp,y1,s1,0,0,0,i,lmbda,nF,nLabels)

                fOrderLmbda0 = lmbda[self._off2+self.getYPairIdx(y1,0,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,0,self._nLabels2)]

                d0s[lp] = d0
                fOrderLmbda0s[lp] = fOrderLmbda0

            for l in range(nLabels):
                if i==0 and not l in startlsSet:
                    ret[l,i] = -inf
                    continue
                (y2,s2) = (self._l2ys[l,0],self._l2ys[l,1])
                l2 = l
                for idx, lp in enumerate(nextls[l]):

                    l1 = lp
                    (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])

                    d = d0s[l1] - fOrderLmbda0s[l1] + \
                    lmbda[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)]

                    valList[idx] = ret[lp,i]+d

                ret[l,i-1] = self.logSumList(valList,len(nextls[l]))
        return ret

    #For now, I don't have log for this, but check if this might be necessary
    cdef predict(self,x_train_t,double[:] lmbda,int nF,int nLabels):

        cdef double d0, d, fOrderLmbda0, val
        cdef int idx, i, curnW, l, l1, lp, l2, s1, y1, s2, y2, nLabels12
        cdef double[:,:] ret
        cdef list prevls
        cdef set startlsSet, endlsSet
        cdef int[:,:] pointers
        cdef int[:] retPath

        startlsSet = set(self._startls)
        endlsSet = set(self._endls)
        curnW = x_train_t[4]

        nLabels12 = self._nLabels1 + self._nLabels2

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
            l1 = l
            (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])


            d = self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,0,0,0,0,lmbda,nF,nLabels)
            ret[l,0] = d
            if not l in startlsSet:
                ret[l,0] = -self._inf

            if curnW == 1 and not l in endlsSet:
                ret[l,0] = -self._inf


        for i in range(1,curnW):
            for l in range(nLabels):
                bestVal = -1
                bestIdx = -1


                l1 = l
                (y1,s1) = (self._l2ys[l1,0],self._l2ys[l1,1])



                d0 = self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,0,0,0,i,lmbda,nF,nLabels)

                fOrderLmbda0 = lmbda[self._off2+self.getYPairIdx(y1,0,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,0,self._nLabels2)]


                if i==curnW-1 and not l in endlsSet:
                    ret[l,i] = -self._inf
                    continue

                for idx,lp in enumerate(prevls[l]):

                    (y2,s2) = (self._l2ys[lp,0],self._l2ys[lp,1])

                    d = d0 - fOrderLmbda0 + \
                    lmbda[self._off2+self.getYPairIdx(y1,y2,self._nLabels1)] + lmbda[self._off3 + self.getYPairIdx(s1,s2,self._nLabels2)]


                    val = ret[lp,i-1]+d
                    if val>bestVal:
                        bestVal = val
                        bestIdx = lp

                ret[l,i] = bestVal
                pointers[l,i] = bestIdx


        retPath[curnW-1] = np.argmax(ret[:,curnW-1])
        for i in range(curnW-1,0,-1):
            retPath[i-1] = pointers[retPath[i],i]
        return retPath


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


    cdef compareVec(self, double[:] l1, double[:] l2):
        cdef int i
        if (l1==None or l2==None):
            return False
        for i in range(len(l1)):
            if (l1[i]!=l2[i]):
                return False
        return True

    cdef copyVec(self, double[:]l1, double[:] l2):
        if (l1==None or len(l1)!=len(l2)):
            l1 = cvarray(shape=(len(l2),), itemsize=sizeof(double), format="d")
        for i in range(len(l2)):
            l1[i] = l2[i]
        return l1

    cdef computell(self,X_train,Y_train,Y_train1,Y_train2,double[:] lmbda,int nF,int nLabels):
        cdef int nSamples, curnW
        cdef double ret
        cdef double[:,:] alphas


        cdef int[:] x_idxes, y_idxes;
        cdef int[:] jidxes
        cdef double[:] vals
        cdef list wIdxes


        nSamples = len(X_train)
        ret = 0

        shouldComputeAlpha = not self.compareVec(self._currentLambda,lmbda)
        if shouldComputeAlpha:
            self._currentLambda = self.copyVec(self._currentLambda, lmbda)

        for i in range(nSamples):

            x_train = X_train[i]
            y_train = Y_train[i]
            y_train1 = Y_train1[i]
            y_train2 = Y_train2[i]

            (x_idxes,y_idxes,vals,wIdxes,curnW) = self.getIdxesSparse(x_train)
            x_train_t = (x_idxes,y_idxes,vals,wIdxes,curnW)

            if (shouldComputeAlpha):
                alphas = self.computeAlphas(x_train_t,lmbda,nF,nLabels,i)
            else:
                alphas = self._currentAlphas[i]


            ret = ret + self.computellOneSample(x_train_t,y_train,y_train1,y_train2,lmbda,alphas,curnW,nF,nLabels)


        ret -= (self._alpha * (np.linalg.norm(lmbda[:self._off2])**2) + self._alpha2 * (np.linalg.norm(lmbda[self._off2:])**2))

        return ret

    cdef computellDer(self,X_train,Y_train,Y_train1,Y_train2,double[:] lmbda,int nF,int nLabels):
        cdef int nSamples, nAllFeatures, i, nW
        cdef double[:] ret
        cdef double[:,:] alphas, betas
        cdef int[:] y_train,y_train1,y_train2


        nSamples = len(X_train)
        nAllFeatures = nF*(self._nLabels1+self._nLabels2) + self._nLabels1**2 + self._nLabels2**2 + self._nLabels1*self._nLabels2 #0-order + 1-orders

        ret = cvarray(shape=(nAllFeatures,), itemsize=sizeof(double), format="d")
        for i in range(nAllFeatures):
            ret[i] = 0

        shouldComputeAlpha = not self.compareVec(self._currentLambda,lmbda)
        if shouldComputeAlpha:
            self._currentLambda = self.copyVec(self._currentLambda, lmbda)

        for i in range(nSamples):
            x_train = X_train[i]
            y_train = Y_train[i]
            y_train1 = Y_train1[i]
            y_train2 = Y_train2[i]

            x_train_t = self.getIdxesSparse(x_train)
            if (shouldComputeAlpha):
                alphas = self.computeAlphas(x_train_t,lmbda,nF,nLabels,i)
            else:
                alphas = self._currentAlphas[i]
            betas = self.computeBetas(x_train_t,lmbda,nF,nLabels)
            nW = x_train.shape[0]
            self.computellDerOneSample(x_train_t,y_train,y_train1,y_train2,lmbda,alphas,betas,nW,nF,nLabels,ret)

        for i in range(self._off2):
            ret[i] -= 2*self._alpha * lmbda[i]
        for i in range(self._off2,len(ret)):
            ret[i] -= 2*self._alpha2 * lmbda[i]
        print "derivative: ",np.array(ret)
        return ret

    cdef computellOneSample(self,x_train_t,int[:] y_train,y_train1,y_train2,double[:] lmbda,double[:,:] alphas,int nW,int nF,int nLabels):

        ret = 0
        for j in range(nW):
            ret += self.computeDotLambdaH(x_train_t,y_train,y_train1,y_train2,j,lmbda,nF,nLabels)


        ret -= self.computePartitionFunction(alphas,nW,nLabels)
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

        cdef list ret
        ret = [x_idxes,y_idxes,vals,wIdxes,curnW]
        return ret



    #The input is X_train[i] for some i: nW*nF for a sentence
    cdef computeDotLambdaH(self,x_train_t,int[:] y_train, int[:] y_train1, int[:] y_train2, int j, double[:] lmbda, int nF, int nLabels):
        cdef int l1,l2
        l1 = y_train[j]
        y1 = y_train1[j]
        s1 = y_train2[j]

        l2 = 0
        y2 = 0
        s2 = 0
        if j!=0:
            l2 = y_train[j-1]
            y2 = y_train1[j-1]
            s2 = y_train2[j-1]

        return self.computeDotLambdaHLabels(x_train_t,l1,y1,s1,l2,y2,s2,j,lmbda,nF,nLabels)

    #The input is X_train[i] for some i: nW*nF for a sentence
    #l1 is the current label, l2 is the previous one
    cdef computeDotLambdaHLabels(self,x_train_t, int l1, int y1, int s1, int l2, int y2, int s2, int j,double[:] lmbda, int nF, int nLabels):

        cdef int[:] x_idxes, y_idxes
        cdef int[:] jidxes
        cdef int curnW, nLabels1, nLabels2, nLabels12
        cdef double[:] vals
        cdef list wIdxes, i2l
        cdef double ret, val
        cdef dict comp_feats

        comp_feats = self._comp_feats
        i2l = self._i2l

        (x_idxes,y_idxes,vals,wIdxes,curnW) = x_train_t
        nLabels1 = self._nLabels1
        nLabels2 = self._nLabels2
        nLabels12 = self._nLabels1 + self._nLabels2

        jidxes = wIdxes[j]

        ret = 0
        for ii in jidxes:
            #BIO features

            val = vals[ii] if vals[ii]!=-1 else comp_feats[y_idxes[ii]](i2l[l1])
            ret += lmbda[y_idxes[ii] + nF * y1] * val
            #sst features
            ret += lmbda[y_idxes[ii] + self._off1 + nF * s1] * val


        if (j>0):
            ret += lmbda[self._off2+self.getYPairIdx(y1,y2,nLabels1)]
            ret += lmbda[self._off3 + self.getYPairIdx(s1,s2,nLabels2)]

        ret += lmbda[self._off4 + self.getYPairIdx(s1,y1,nLabels1)]

        return ret

    cdef getYPairIdx(self,int l1,int l2,int nLabels):
        return l1*nLabels+l2


    cdef getLegalNextLabels(self,i2ls):
        cdef list nextls, prevls
        cdef int[:] startls, endls
        cdef int l, i

        nextls =[[] for i in range(len(i2ls))]
        prevls = [[] for i in range(len(i2ls))]
        for l in range(len(i2ls)):
            for l2 in range(len(i2ls)):
                if legalTagBigramForLogistic(i2ls[l],i2ls[l2],'NO_SINGLETON_B'):
                    nextls[l].append(l2)
                if legalTagBigramForLogistic(i2ls[l2],i2ls[l],'NO_SINGLETON_B'):
                    prevls[l].append(l2)


        ends = []
        starts = []
        for l in range(len(i2ls)):
            if legalTagBigramForLogistic(i2ls[l],None,'NO_SINGLETON_B'):
                ends.append(l)
            if legalTagBigramForLogistic(None, i2ls[l],'NO_SINGLETON_B'):
                starts.append(l)


        endls = np.array(ends, dtype='int32')
        startls = np.array(starts, dtype='int32')

        return (prevls,nextls, startls, endls)



    cdef extractL1L2s(self, Y_train,_labels):
        _labels1 = features.SequentialStringIndexer()
        _labels2 = features.SequentialStringIndexer()

        for y_train in Y_train:
            for y in y_train:
                s = _labels.strings[y].split("-")
                l1 = s[0]
                if len(s)>1:
                    l2 = s[1]
                else:
                    l2 = '0'

                _labels1.add(l1)
                _labels2.add(l2)

        _labels1.freeze()
        _labels2.freeze()

        Y_train1 = []
        Y_train2 = []


        for y_train in Y_train:
            y_train1 = cvarray(shape=(len(y_train),), itemsize=sizeof(int), format="i")
            y_train2 = cvarray(shape=(len(y_train),), itemsize=sizeof(int), format="i")

            for idx,y in enumerate(y_train):
                s = _labels.strings[y].split("-")
                l1 = s[0]
                if len(s)>1:
                    l2 = s[1]
                else:
                    l2 = '0'
                y_train1[idx] = _labels1.s2i[l1]
                y_train2[idx] = _labels2.s2i[l2]
            Y_train1.append(y_train1)
            Y_train2.append(y_train2)
        return (Y_train1,Y_train2,_labels1,_labels2)



    cdef getl2ys(self, int nLabels,_labels, _labels1, _labels2):
        cdef int[:,:] ret
        ret = cvarray(shape=(nLabels,2), itemsize=sizeof(int), format="i")
        for l in range(nLabels):
            (y,s) = getYSFromL(l, _labels,_labels1,_labels2)
            ret[l,0] = y
            ret[l,1] = s
        return ret



    cdef extractXYs(self, trainingData):

        cdef int[:] Ycur
        cdef dict comp_feats
        cdef double[:] valsArr
        Y = []
        YMap = {}
        nS = nW = 0
        comp_feats = {}

        _features = features.SequentialStringIndexer(cutoff=self._cutoff)
        _labels = features.SequentialStringIndexer()
        _featureNames = {}


        for sent,o0FeatsEachToken in trainingData:
            nS += 1
            if (nS==self._maxSentence):
                break

            for w,o0Feats in zip(sent,o0FeatsEachToken):
                nW += 1

                _labels.add(w.gold)

                indices = [i for (i,v) in o0Feats.items()]
                for i in indices:
                    _features.add(str(i))
                    _featureNames[str(i)] = str(o0Feats._set._indexer[i])


        _labels.freeze()

        l2i = _labels.s2i
        i2l = _labels.strings


        _features.freeze()
        nF = len(_features.strings)
        print "num features:", nF


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
            if (nS==self._maxSentence):
                break

            rowsCur = []
            colsCur = []
            valsCur = []
            Ycur = cvarray(shape=(len(sent),), itemsize=sizeof(int), format="i")

            wIdxCur = 0

            for w,o0Feats in zip(sent,o0FeatsEachToken):

                y = l2i[w.gold]
                Y.append(y)
                Ycur[wIdxCur] = y
                indices = [i for (i,v) in o0Feats.items()]

                for (i,v) in o0Feats.items():
                    print ("i:",i,"name: ",str(o0Feats._set._indexer[i]),"v:",v)

                    if (str(i) in i2s_set):
                        featIndex = s2i[str(i)]
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

            valsArr = cvarray(shape=(len(valsCur),), itemsize=sizeof(double), format="d")
            for ii in range(len(valsCur)):
                valsArr[ii] = valsCur[ii]
            X_cur_sparse = csr_matrix( (valsArr,(rowsCur,colsCur)), shape=(wIdxCur,nF))
            X_train_sparse.append((X_cur_sparse))
            Y_train.append(Ycur)



        nW= wI
        print "nW: ", nW

        nLabels = len(l2i)

        print X_train
        print Y_train

        cdef int[:] starts, ends

        X_train_sparse = np.array(X_train_sparse)
        Y_train = np.array(Y_train)
        print "nlabels: ", len(i2l)
        (prevls,nextls, starts, ends) = self.getLegalNextLabels(i2l)

        (Y_train1,Y_train2,_labels1,_labels2) = self.extractL1L2s(Y_train,_labels)

        l2ys = self.getl2ys(nLabels,_labels, _labels1, _labels2)

        return (_features, _labels, _labels1,_labels2, X_train_sparse,Y_train,Y_train1,Y_train2,nF,nLabels,prevls,nextls,starts, ends,i2l,i2s,l2ys, _featureNames, comp_feats)


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
            Ycur = cvarray(shape=(len(sent),), itemsize=sizeof(int), format="i")

            wIdxCur = 0

            nTokens = len(sent)

            for w,o0Feats in zip(sent,o0FeatsEachToken):

                #This is just for debugging with fewer sentences
                y = -1
                if w.gold in i2l:
                    y = l2i[w.gold]
                Ycur[wIdxCur] = y

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

            preds = self.predict(x_train_t,self._lmbda,nF,nLabels)

            for j in range(wIdxCur):
                pred = i2l[preds[j]]

                sent[j] = sent[j]._replace(prediction=pred)
                print "predict: ", pred.encode('utf-8'), ", gold ", sent[j].__str__().encode('utf-8')

                if (preds[j]==Ycur[j]):
                    correctCount += 1



            sent.updatedPredictions_without_assert()
            if (outFile is not None):
                outFile.write(sent.tagsStr()+"\n\n")
        print "acc: ", (float)(correctCount)/all
