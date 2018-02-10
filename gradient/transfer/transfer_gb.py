import numpy as np
import math
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)

def reversed_cmp(x, y):
   if x > y:
       return -1
   if x < y:
       return 1
   return 0

def Ronehot(seq):
    res = np.zeros((23*4), dtype=np.uint8)
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    for ii, char in enumerate(CHARS):
        res[ii*seqlen:(ii+1)*seqlen][arr == char] = 1
    ms=res.reshape(4,seqlen).T
    # ap=np.zeros((4,23))
    # for i in range(1,5):
    #     ap[i-1]=Oonehot(f[i])
    # ap=ap.T
    # for i in range(0,23):
    #     for j in range(0,4):
    #         ms[i][j+4]=ap[i][j]
    ms=ms.reshape(1,seqlen*4)
    return ms

#LFILE='hct116.episgt' 
#LCNT=4239
#LFILE='hek293t.episgt' 
#LCNT=4666
#LFILE='hela.episgt' 
#LCNT=8101
#LFILE='hl60.episgt'
#LCNT=2076

def Transfer(LFILE,LCNT,RFILE,RCNT):
    ff=open(LFILE,'r')
    idx=0
    LfRNA=np.zeros((LCNT,23*4))
    Llabel=np.zeros((LCNT,))
    for line in ff:
        f=line.split('\t')
        if(int(f[1])==1):
            Llabel[idx]=1
        else:
            Llabel[idx]=-1
        LfRNA[idx]=Ronehot(f[0])
        idx+=1

    LX_train, LX_test, Ly_train, Ly_test = train_test_split(LfRNA, Llabel, test_size=0.2, random_state=0)
    #print(np.shape(LX_train), np.shape(Ly_train), np.shape(LX_test), np.shape(Ly_test))

    ff=open(RFILE,'r')
    idx=0
    RfRNA=np.zeros((RCNT,23*4))
    Rlabel=np.zeros((RCNT,))
    for line in ff:
        f=line.split('\t')
        if(int(f[1])==1):
            Rlabel[idx]=1
        else:
            Rlabel[idx]=-1
        RfRNA[idx]=Ronehot(f[0])
        idx+=1

    RX_train, RX_test, Ry_train, Ry_test = train_test_split(RfRNA, Rlabel, test_size=0.2, random_state=0)
    #print(np.shape(RX_train), np.shape(Ry_train), np.shape(RX_test), np.shape(Ry_test))

    clf2 = GradientBoostingClassifier().fit(LfRNA, Llabel)
    print("train:",LFILE,"  test:",RFILE)
    print("train size:",np.shape(Llabel),"  test size:",np.shape(Rlabel))
    #print(clf2.score(RfRNA, Rlabel))
    pred=clf2.predict_proba(RfRNA)
    prob=np.zeros((np.shape(Rlabel)[0],))
    labl=np.zeros((np.shape(Rlabel)[0],))
    testcnt=0
    for xx in pred:
        xl=int(Rlabel[testcnt])
        if(xl==-1):
            xl=0
        xp=np.argmax(xx)
        prob[testcnt]=xx[xp]
        labl[testcnt]=xl
        testcnt+=1
    #print(prob)
    #print(labl)
    print('acc: ',clf2.score(RfRNA, Rlabel))
    print('auc: ',roc_auc_score(labl, prob))
