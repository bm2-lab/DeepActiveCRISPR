import tensorflow as tf
import sonnet as snt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import csv
from sklearn.ensemble import GradientBoostingClassifier

CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)

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

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
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


channel_size = [8, 32, 64, 64, 256, 256, 512, 512, 1024, 2]

betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]) ) for i in range(1, len(channel_size))]
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

encoder = [None, e1, e2, e3, e4, e5, e6, e7, e8, e9]
encoder_bn_l = [None, ebn1l, ebn2l, ebn3l, ebn4l, ebn5l, ebn6l, ebn7l, ebn8l, ebn9l]

inputs_l = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 8])
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

eta = 0.01
with tf.control_dependencies(tf.get_collection('update_ops')):
    train_op = tf.train.AdamOptimizer(eta).minimize(loss)


def Ronehot(f):
    seq=f[0]
    res = np.zeros((23*8), dtype=np.uint8)
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    for ii, char in enumerate(CHARS):
        res[ii*seqlen:(ii+1)*seqlen][arr == char] = 1
    ms=res.reshape(8,seqlen).T
    for i in range(1,5):
        seq=f[i]
        idx=0
        for ch in seq:
            if(ch=='N'):
                ms[idx][4+i-1]=0
            else:
                ms[idx][4+i-1]=1
            idx+=1
    return ms

#LFILE='hct116.episgt' 
#LCNT=4239
#LFILE='hek293t.episgt' 
#LCNT=4666
#LFILE='hela.episgt' 
#LCNT=8101
#LFILE='hl60.episgt'
#LCNT=2076
#LFILE='label.dat'
#LCNT=19082
LFILE='all_tr.episgt'
LCNT=12121
TFILE='all_te.episgt'
TCNT=3621


ff=open(LFILE,'r')
idx=0
X_train=np.zeros((LCNT,1,23,8))
y_train=np.zeros((LCNT,2))
for line in ff:
    f=line.split('\t')
    y_train[idx][int(f[5])]=1
    X_train[idx][0]=Ronehot(f)
    idx+=1

ff=open(TFILE,'r')
idx=0
X_test=np.zeros((TCNT,1,23,8))
y_test=np.zeros((TCNT,2))
for line in ff:
    f=line.split('\t')
    y_test[idx][int(f[5])]=1
    X_test[idx][0]=Ronehot(f)
    idx+=1

# ff=open(LFILE,'r')
# idx=0
# fRNA=np.zeros((LCNT,1,23,8))
# label=np.zeros((LCNT,2))
# for line in ff:
#     f=line.split('\t')
#     label[idx][int(f[5])]=1
#     fRNA[idx][0]=Ronehot(f)
#     idx+=1

#X_train, X_test, y_train, y_test = train_test_split(fRNA, label, test_size=0.2, random_state=0, stratify = label)

trainDat=DataSet(X_train,y_train)
batch_size = 32
num_epochs = 100
num_examples = np.shape(y_train)[0]
i_iter=0
num_iter = (num_examples/batch_size) * num_epochs 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

x_test=np.zeros((np.shape(X_test)[0],1,23,8))
i_cnt=0
for xx in X_test:
    x_test[i_cnt][0]=xx
    i_cnt+=1
print i_cnt

prob=hl_last
correct_prediction = tf.equal(tf.argmax(logits_l, 1), tf.argmax(outputs, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
propred = sess.run(prob, feed_dict={inputs_l: x_test, training: False})

for i in tqdm(range(i_iter, num_iter)):
    images, labels = trainDat.next_batch(batch_size)
    sess.run(train_op, feed_dict={inputs_l: images, outputs: labels, training: True})
    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        epoch_n = i/(num_examples/batch_size)
        with open('train_log', 'ab') as train_log:
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            acc=sess.run([accuracy], feed_dict={inputs_l: X_test, outputs: y_test, training: False})
            log_i = [epoch_n] + [acc]
            train_log_w.writerow(log_i)

testcnt=0
prob=np.zeros((np.shape(y_test)[0],))
labl=np.zeros((np.shape(y_test)[0],))
for xx in propred:
    print (xx[0][0][0],xx[0][0][1],np.argmax(xx[0][0]),np.argmax(y_test[testcnt]))
    prob[testcnt]=xx[0][0][1]
    labl[testcnt]=np.argmax(y_test[testcnt])
    testcnt+=1
print(prob)
print(labl)
print(np.shape(prob),np.shape(labl))
print(LFILE, (np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)))
print (sess.run(accuracy, feed_dict={inputs_l: X_test, outputs: y_test, training: False}))
print roc_auc_score(labl, prob)
