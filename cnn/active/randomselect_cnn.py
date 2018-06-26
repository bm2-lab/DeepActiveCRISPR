import numpy as np
import math
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import csv
import pdb
import random
import sys
 
#dataset
_LFILE=['hct116.episgt','hek293t.episgt','hela.episgt','hl60.episgt']
_LCNT=[4239,4666,8101,2076]
_BMODEL=['ex_hct116_14843.episgt_model.ckpt',
        'ex_hek293t_14416.episgt_model.ckpt',
        'ex_hela_10981.episgt_model.ckpt',
        'ex_hl60_17006.episgt_model.ckpt']
DataNo=int(sys.argv[1])
LFILE=_LFILE[DataNo]
LCNT=_LCNT[DataNo]
BMODEL=_BMODEL[DataNo]
drawcolor='c'

#adjustable paramaters
##for CNN
batch_size = 32         #batch size  
num_epochs = 40         #num of epochs 
start_eta = 0.00004     #learning rate 
decay_lr = 1            #decay ratio of learning rate in each epoch
##for active learning
#alpha=1.0/4             #proportion of the patch used for majority selection
b=16                     #num of elements added to labeled set each iteration 
#lD=1                    #weight of Entropy
#lE=1                    #weight of Diversity

LOGDIR='./log/'+LFILE+'_random_log.txt'
LFILE='../../dataset/'+LFILE
BMODEL='../premodel/'+BMODEL

def print2f(MSG):
    print(MSG)
    lf=open(LOGDIR,'a')
    print >> lf,MSG
    lf.close()

def Ronehot(seq):
    res = np.zeros((23*4), dtype=np.uint8)
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    for ii, char in enumerate(CHARS):
        res[ii*seqlen:(ii+1)*seqlen][arr == char] = 1
    ms=res.reshape(4,seqlen).T
    return ms

def reversed_cmp(x, y):
   if x > y:
       return -1
   if x < y:
       return 1
   return 0

CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)

ff=open(LFILE,'r')
idx=0
fRNA=np.zeros((LCNT,1,23,4))
label=np.zeros((LCNT,2))
for line in ff:
    f=line.split('\t')
    label[idx][int(f[1])]=1
    fRNA[idx][0]=Ronehot(f[0])
    idx+=1

X_train, X_test, y_train, y_test = train_test_split(fRNA, label, test_size=0.2, random_state=0, stratify = label)
print2f((np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)))

inttrain_x=np.zeros((0,1,23,4))               #initial training set in active learning - X
inttrain_y=np.zeros((0,2))                    #initial training set in active learning - y

TRAINLEN=np.shape(X_train)[0]                 #size of training set     
INTLEN=np.shape(inttrain_x)[0]
AUGLEN=TRAINLEN*16                            #size of augmented training set

Uset=set()
Lset=set()
for i in range(0,INTLEN):
    Lset.add(i)

for i in range(INTLEN,TRAINLEN):
    Uset.add(i)

train_x=np.zeros((AUGLEN,1,23,4))             #augmented training set - X
train_y=np.zeros((AUGLEN,2))                  #augmented training set - y
train_l=np.zeros((AUGLEN,1))                  #augmented training set - label of each element
patch=[set() for x in range(0,TRAINLEN)]

for i in range(0,TRAINLEN):
    sample=X_train[i][0].copy()
    R1=np.zeros((4))
    R2=np.zeros((4))
    for j in range(0,4):
        R1[j]=1
        for k in range(0,4):
            R2[k]=1
            #RR=np.concatenate((R1,R2))
            sample[0]=R1.copy()
            sample[1]=R2.copy()
            train_x[i*16+j*4+k][0]=sample.copy()
            train_y[i*16+j*4+k]=y_train[i]
            train_l[i*16+j*4+k]=i
            #print("AUG ",i*16+j*4+k,i)
            patch[i].add(i*16+j*4+k)
            R2[k]=0
        R1[j]=0

print2f ((TRAINLEN,AUGLEN,INTLEN))
print2f ((np.shape(X_train)[0],np.shape(train_x)[0],np.shape(inttrain_x)[0]))
print2f ((patch[1]))


class DataSet(object):

    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape,labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            #images = images.astype(np.float32)
            #images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(23*4)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


### definition of CNN ###
channel_size = [4, 32, 32, 64, 256, 256, 512, 512, 1024, 2]

#betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in range(1, len(channel_size))]
betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_'+str(i)) for i in range(1, len(channel_size))]
gamma = tf.Variable(1.0 * tf.ones(channel_size[-1]), name='gamma')

e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
ebn1l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_1l')
e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
ebn2l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_2l')
e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
ebn3l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_3l')
e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
ebn4l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_4l')
e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
ebn5l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_5l')
e6 = snt.Conv2D(channel_size[6], kernel_shape=[1, 3], stride=2, name='e_6')
ebn6l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_6l')
e7 = snt.Conv2D(channel_size[7], kernel_shape=[1, 3], name='e_7')
ebn7l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_7l')
e8 = snt.Conv2D(channel_size[8], kernel_shape=[1, 3], padding='VALID', name='e_8')
ebn8l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_8l')
e9 = snt.Conv2D(channel_size[9], kernel_shape=[1, 1], name='e_9')
ebn9l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_9l')
e0 = snt.Conv2D(channel_size[9], kernel_shape=[1, 1], name='e_0')
ebn0l = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_0l')
ea = snt.Conv2D(channel_size[9], kernel_shape=[1, 1], name='e_a')
ebnal = snt.BatchNorm(decay_rate=0.99, offset=False, name='ebn_al')

encoder = [None, e1, e2, e3, e4, e5, e6, e7, e8, e9]
encoder_bn_l = [None, ebn1l, ebn2l, ebn3l, ebn4l, ebn5l, ebn6l, ebn7l, ebn8l, ebn9l]

inputs_l = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 4])
outputs_raw = tf.placeholder(tf.uint8, shape=[None])
outputs = tf.one_hot(outputs_raw, depth=2)
training = tf.placeholder(dtype=tf.bool)

hl0 = inputs_l
l_lst = [hl0]
hl_lst = [hl0]

for i in range(1, len(channel_size) - 1):
    hl_pre = hl_lst[i - 1]
    pre_l = encoder[i](hl_pre)
    l = encoder_bn_l[i](pre_l, training)
    hl = tf.nn.relu(l + betas[i])
    l_lst.append(l)
    hl_lst.append(hl)

hl_m1 = hl_lst[-1]
pre_l_last = encoder[-1](hl_m1)
l_last = encoder_bn_l[-1](pre_l_last, training)
l_last = gamma * l_last + betas[-1]
hl_last = tf.nn.softmax(l_last)
l_lst.append(l_last)
hl_lst.append(hl_last)

logits_l = tf.squeeze(l_last)
sig_l = tf.squeeze(hl_last)[:, 1]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=logits_l))

learning_rate = tf.Variable(start_eta, trainable=False)
with tf.control_dependencies(tf.get_collection('update_ops')):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)



###pre-training### using X_base,y_base -- X_test,y_test
saver = tf.train.Saver()
model_path=BMODEL
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess,model_path)

x_test=np.zeros((np.shape(X_test)[0],1,23,4))
i_cnt=0
for xx in X_test:
    x_test[i_cnt][0]=xx
    i_cnt+=1
print2f(i_cnt)

prob=hl_last
correct_prediction = tf.equal(tf.argmax(logits_l, 1), tf.argmax(outputs, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
propred = sess.run(prob, feed_dict={inputs_l: x_test, training: False})

testcnt=0
prob=np.zeros((np.shape(y_test)[0],))
labl=np.zeros((np.shape(y_test)[0],))
for xx in propred:
    #print (xx[0][0][0],xx[0][0][1],np.argmax(xx[0][0]),np.argmax(y_test[testcnt]))
    prob[testcnt]=xx[0][0][np.argmax(xx[0][0])]
    labl[testcnt]=np.argmax(y_test[testcnt])
    testcnt+=1
#print(prob)
#print(labl)
print2f (('pre-train', BMODEL))
print2f (sess.run(accuracy, feed_dict={inputs_l: X_test, outputs: y_test, training: False}))
auc_test = roc_auc_score(labl, sess.run(sig_l, feed_dict={inputs_l: X_test, training: False}))
print2f(auc_test)


# clf = GradientBoostingClassifier().fit(inttrain_x, inttrain_y)
# print (("init: ",clf.score(X_test, y_test)))
# clf2 = GradientBoostingClassifier().fit(X_train, y_train)
# print (("complete: ",clf2.score(X_test, y_test)))


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
    Rm=random.sample(Uset,b)
    for elm in Rm:
        Lset.add(elm)
        Uset.remove(elm)
        inttrain_x=np.concatenate((inttrain_x, [X_train[elm]]), axis=0)
        inttrain_y=np.concatenate((inttrain_y, [y_train[elm]]), axis=0)
        INTLEN+=1
    print2f ((np.shape(inttrain_x)[0],len(Lset),len(Uset)))

    ###fine-tune### using inttrain_x,inttrain_y -- X_test,y_test
    trainDat=DataSet(inttrain_x,inttrain_y)
    num_examples = np.shape(inttrain_y)[0]
    i_iter=0
    num_iter = (num_examples/batch_size) * num_epochs 

    prob=hl_last
    correct_prediction = tf.equal(tf.argmax(logits_l, 1), tf.argmax(outputs, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
    propred = sess.run(prob, feed_dict={inputs_l: x_test, training: False})

    #eta=start_eta
    #sess.run(learning_rate.assign(eta))
    for i in tqdm(range(i_iter, num_iter)):
        images, labels = trainDat.next_batch(batch_size)
        sess.run(train_op, feed_dict={inputs_l: images, outputs: labels, training: True})
        #eta=eta*decay_lr
        #sess.run(learning_rate.assign(eta))
        if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
            epoch_n = i/(num_examples/batch_size)

    testcnt=0
    prob=np.zeros((np.shape(y_test)[0],))
    labl=np.zeros((np.shape(y_test)[0],))
    for xx in propred:
        #print (xx[0][0][0],xx[0][0][1],np.argmax(xx[0][0]),np.argmax(y_test[testcnt]))
        prob[testcnt]=xx[0][0][np.argmax(xx[0][0])]
        labl[testcnt]=np.argmax(y_test[testcnt])
        testcnt+=1
    #print(prob)
    #print(labl)
    print2f (('finetune:', LFILE, np.shape(inttrain_x), np.shape(inttrain_y), np.shape(X_test), np.shape(y_test)))
    print2f (sess.run(accuracy, feed_dict={inputs_l: X_test, outputs: y_test, training: False}))

    auc_test = roc_auc_score(labl, sess.run(sig_l, feed_dict={inputs_l: X_test, training: False}))
    print2f(auc_test)
    print2f("----------------------------")
    
    print2f (("iter: ",IT,auc_test))
    ACC.append(auc_test)
    ITR.append(IT)
    LAB.append(len(Lset))

plt.plot(LAB,ACC,drawcolor)
#plt.plot(LAB,ACC,'b*')
plt.xlabel('Num of labels')
plt.ylabel('AUC')
plt.ylim(0.5,1.0)
plt.title(LFILE)
#plt.show()
plt.savefig(LOGDIR.replace(".txt",".jpg"))


print2f("______________data_______________")
print2f("LFILE=")
print2f(_LFILE[DataNo])
print2f("LAB=")
print2f(LAB)
print2f("ACC=")
print2f(ACC)



