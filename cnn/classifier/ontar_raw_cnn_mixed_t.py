import tensorflow as tf
import sonnet as snt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from progress.bar import Bar

def gnds(x, y, batch=32, shuffle=True, allow_smaller_final_batch=False):
    num_examples = len(y)
    idxes = np.arange(num_examples)
    N = num_examples // batch
    res = num_examples % batch
    while True:
        if shuffle:
            np.random.shuffle(idxes)
        for i in range(N):
            ib = idxes[i * batch: (i + 1) * batch]
            yield (x[ib], y[ib])
        if res != 0:
            if allow_smaller_final_batch:
                ib = idxes[-res:]
                yield (x[ib], y[ib])

channels = 4


channel_size = [channels, 32, 64, 64, 256, 256, 512, 512, 1024, 2]

betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i])) for i in range(1, len(channel_size))]
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

inputs_l = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 4])
outputs_raw = tf.placeholder(tf.uint8, shape=[None])
outputs = tf.one_hot(outputs_raw, depth=2)
training = tf.placeholder(dtype=tf.bool)

# clean encoder
hl0 = inputs_l
l_lst = [hl0]
hl_lst = [hl0]

for i in range(1, len(channel_size) - 1):
    hl_pre = hl_lst[i - 1]
    pre_l = encoder[i](hl_pre)
    l = encoder_bn_l[i](pre_l, training, test_local_stats=False)
    hl = tf.nn.relu(l + betas[i])
    l_lst.append(l)
    hl_lst.append(hl)

hl_m1 = hl_lst[-1]
pre_l_last = encoder[-1](hl_m1)
l_last = encoder_bn_l[-1](pre_l_last, training, test_local_stats=False)
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



xtr, ytr, xte, yte = joblib.load('all.pkl')
xtr = xtr[:,:,:,:channels]
xte = xte[:,:,:,:channels]
num_examples = len(ytr)
batch_size_l = 32
labeled_gn = gnds(xtr, ytr, batch=batch_size_l)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

num_steps_per_epoch = num_examples // batch_size_l

n_epochs = 30

for epoch in range(n_epochs):
    bar = Bar('Epoch '+str(epoch+1), max=num_steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
    for i in range(num_steps_per_epoch):
        xtr_l, ytr = next(labeled_gn)
        ytr = ytr.ravel()
        feed_dict_train = {inputs_l: xtr_l, outputs_raw: ytr, training: True}
        sess.run(train_op, feed_dict=feed_dict_train)
        bar.next()
    bar.finish()
    auc_train = roc_auc_score(ytr, sess.run(sig_l, feed_dict={inputs_l: xtr_l, training: False}))
    loss_value = sess.run(loss,
                          feed_dict={inputs_l: xtr_l,
                                     outputs_raw: ytr, training: False})
    auc_test = roc_auc_score(yte, sess.run(sig_l, feed_dict={inputs_l: xte, training: False}))
    test_loss_value = sess.run(loss,
                               feed_dict={inputs_l: xte, training: False,
                                          outputs_raw: yte})

    print('Epoch ',epoch+1)
    print('Training:')
    print('loss: ',loss_value)
    print('clean loss: ',loss_value)
    print('auc: ',auc_train)
    print('Testing:')
    print('clean loss: ',test_loss_value)
    print('auc: ',auc_test)

