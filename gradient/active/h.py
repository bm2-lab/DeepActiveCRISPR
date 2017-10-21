import numpy as np
import math
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 


# alpha=1.0/4
# b=8
# lE=1
# lD=1

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
#_,LFILE,LCNT=argv
#LCNT=int(LCNT)


def ActLearning(LFILE,LCNT,alpha,b,lD,lE,color,idn):
    def print2f(MSG):
        lf=open(LFILE+'_out'+idn+'.txt','a')
        print >> lf,MSG
        lf.close()

    ff=open(LFILE,'r')
    idx=0
    fRNA=np.zeros((LCNT,23*4))
    label=np.zeros((LCNT,))
    for line in ff:
        f=line.split('\t')
        if(int(f[5])==1):
            label[idx]=1
        else:
            label[idx]=-1
        fRNA[idx]=Ronehot(f[0])
        idx+=1


    X_train, X_test, y_train, y_test = train_test_split(fRNA, label, test_size=0.2, random_state=0)
    print2f((np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)))

    TRAINLEN=np.shape(X_train)[0]
    INTLEN=int(TRAINLEN*0.01)

    aa=np.split(X_train,[TRAINLEN-INTLEN,TRAINLEN-INTLEN])
    X_train=aa[0]
    inttrain_x=aa[2]
    aa=np.split(y_train,[TRAINLEN-INTLEN,TRAINLEN-INTLEN])
    y_train=aa[0]
    inttrain_y=aa[2]

    TRAINLEN=np.shape(X_train)[0]
    INTLEN=np.shape(inttrain_x)[0]
    AUGLEN=TRAINLEN*16

    Uset=set()
    Lset=set()
    for i in range(0,INTLEN):
        Lset.add(i)
    for i in range(INTLEN,TRAINLEN):
        Uset.add(i)

    train_x=np.zeros((AUGLEN,23*4))
    train_y=np.zeros((AUGLEN,))
    train_l=np.zeros((AUGLEN,))
    patch=[set() for x in range(0,TRAINLEN)]

    for i in range(0,TRAINLEN):
        sample=X_train[i]
        R1=np.zeros((4))
        R2=np.zeros((4))
        for j in range(0,4):
            R1[j]=1
            for k in range(0,4):
                R2[k]=1
                RR=np.concatenate((R1,R2))
                for x in range(0,8):
                    sample[x]=RR[x]
                train_x[i*16+j*4+k]=sample
                train_y[i*16+j*4+k]=y_train[i]
                train_l[i*16+j*4+k]=i
                patch[i].add(i*16+j*4+k)
                R2[k]=0
            R1[j]=0

    print2f ((TRAINLEN,AUGLEN,INTLEN))
    print2f ((np.shape(X_train)[0],np.shape(train_x)[0],np.shape(inttrain_x)[0]))
    print2f ((patch[0]))

    clf = GradientBoostingClassifier().fit(inttrain_x, inttrain_y)
    print2f (("init: ",clf.score(X_test, y_test)))
    clf2 = GradientBoostingClassifier().fit(X_train, y_train)
    print2f (("complete: ",clf2.score(X_test, y_test)))


    #for i in range(10,20):
    #    print (clf.predict(X_test[i]), clf.predict_proba(X_test[i])[0][1], clf.predict_log_proba(X_test[i])[0][1], math.log(clf.predict_proba(X_test[i])[0][1]), y_test[i])

    eps=np.spacing(1)
    ITER=int(TRAINLEN/b)
    patchsize=16
    predpatch=[0.0 for x in range(0,patchsize)]
    ACC=[]
    ITR=[]
    LAB=[]

    for IT in range(0,ITER):
        if(INTLEN+b>TRAINLEN):
            print2f(("OUT OF RANGE "))
            break
        Ulen=len(Uset)
        Rm=[]
        for x in Uset:		#sample x
            meanpred=0.0
            pn=0
            for xp in patch[x]:	#all patches of x
                predpatch[pn]=clf.predict_proba(train_x[xp])[0][1]
                meanpred+=predpatch[pn]
                pn+=1
            meanpred/=patchsize
            predpatch.sort()
            plen=int(round(alpha*patchsize))
            if(meanpred>0.5):
                Sp=predpatch[patchsize-plen:patchsize]
            else:
                Sp=predpatch[0:plen]
            Ex=0.0                                      #Entropy of x
            Dx=0.0                                      #Diversity of x
            for ip,p in enumerate(Sp):
                for iq,q in enumerate(Sp):
                    if(p==0.00):
                        p=p+eps
                    if(p==1.00):
                        p=p-eps
                    if(q==0.00):
                        q=q+eps
                    if(q==1.00):
                        q=q-eps
                    if(ip!=iq):
                        Dx+=(p-q)*np.log(p/q)
                        Dx+=((1-p)-(1-q))*np.log((1-p)/(1-q))
                    else:
                        Ex-=p*np.log(p)
                        Ex-=(1-p)*np.log(1-p)
            Rsum=lE*Ex+lD*Dx
            Rm.append((Rsum,x))
        Rm=sorted(Rm,reversed_cmp)
        for x in range(0,b):
            elm=int(Rm[x][1])
            Lset.add(elm)
            Uset.remove(elm)
            inttrain_x=np.concatenate((inttrain_x, [X_train[elm]]), axis=0)
            inttrain_y=np.concatenate((inttrain_y, [y_train[elm]]), axis=0)
            INTLEN+=1
        print2f ((np.shape(inttrain_x)[0],len(Lset),len(Uset)))
        clf = GradientBoostingClassifier().fit(inttrain_x, inttrain_y)
        res=clf.score(X_test, y_test)
        print2f (("iter: ",IT,res))
        ACC.append(res)
        ITR.append(IT)
        LAB.append(len(Lset))

    plt.plot(LAB,ACC,color)
    #plt.plot(LAB,ACC,'b*')
    plt.xlabel('Num of labels')
    plt.ylabel('Accuracy')
    plt.ylim(0.5,1.0)
    plt.title(LFILE)
    plt.show()



