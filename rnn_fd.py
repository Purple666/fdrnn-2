import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn 

import numpy as np
import math 
import matplotlib.pyplot as plt
import sys
from data_loader import data_feeder

import os
N_LAYERS = 4
LAYER_SIZE=1024
SEQ_LEN=512
P_KEEP = 1.0
ALPHA_SIZE=10
BATCH_SIZE =16

AMOUNT_L=1
learning_rate=0.001

dim_x=3
dim_y=2



p_keep = tf.placeholder(tf.float32, name='p_keep')  # dropout parameter
batch_size = tf.placeholder(tf.int32, name='batch_size')
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
seq_len=tf.placeholder(tf.int32, name="seq_len")
X=tf.placeholder(tf.float32,shape=[None, None, dim_x], name='X') 

Y=tf.placeholder(tf.float32,shape=[None, None, dim_y], name='Y') 

Hin = tf.placeholder(tf.float32, [N_LAYERS,2, None, LAYER_SIZE], name='Hin')
rnn_tuple_state = tuple(
     [tf.nn.rnn_cell.LSTMStateTuple(Hin[idx,0], Hin[idx,1])
     for idx in range(N_LAYERS)]
)

saved_states=tf.Variable(tf.constant(np.zeros([N_LAYERS,2,BATCH_SIZE, LAYER_SIZE]),dtype=tf.float32), name='saved_states', dtype=tf.float32)
cells = [rnn.LSTMCell(LAYER_SIZE, state_is_tuple=True) for _ in range(N_LAYERS)] 
multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)
Ynn, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=rnn_tuple_state)
sv_op=saved_states.assign(Hin)

H=tf.identity(H, name='H')
Ynn0=Ynn[:, :, 0]
Ynn1=Ynn[:, :, 1]


Y_predict_amount=tf.exp(tf.multiply(Ynn0,tf.constant(20, tf.float32)), "Y_predict_amount")


Y_predict_angle=tf.multiply(Ynn1, tf.constant(math.pi, tf.float32), "Y_predict_angle")
Y_predict=tf.concat([tf.reshape(Y_predict_amount, [batch_size,seq_len, 1]),tf.reshape(Y_predict_angle, [batch_size,seq_len, 1])] , 2, name="Y_predict")

Y_amount=Y[:, :, 0]
Y_angle=Y[:, :, 1]
Y_amount_biased=tf.add(Y_amount, tf.constant(0.000000001, tf.float32))
Y_amount_log=tf.log(Y_amount_biased)


Y_predict_amount_biased=tf.add(Y_predict_amount, tf.constant(0.000000001, tf.float32))
Y_amount_log_predict=tf.log(Y_predict_amount_biased)

amount_L=tf.constant(AMOUNT_L, tf.float32)
loss=tf.multiply(tf.losses.mean_squared_error(labels=Y_amount_log, predictions=Y_amount_log_predict), amount_L)+tf.losses.mean_squared_error(labels=Y_angle, predictions=Y_predict_angle)

train_step = tf.train.AdamOptimizer(lr).minimize(loss)


batchloss=loss


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batchlosses=[]
init_op = tf.variables_initializer([saved_states,])



prev_epoch=0
sb=0
ini_state = np.zeros([N_LAYERS,2,BATCH_SIZE, LAYER_SIZE])
state=ini_state
for x, y, epoch, batch_num in data_feeder( BATCH_SIZE, SEQ_LEN, epoch_num=10):
    _, y, state, n_Y_predict, n_loss,_ , fed_init,_= sess.run([train_step, Y, H, Y_predict, batchloss, saved_states, Hin,sv_op], feed_dict={X: x, Y: y, Hin: state, lr: learning_rate, p_keep: P_KEEP, batch_size: BATCH_SIZE, seq_len:SEQ_LEN})
    print("________________________________________________")
    print("epoch")
    print(epoch)
    print("Y_predict")
    print(n_Y_predict)
    print("y")
    print(y)
    print("loss")
    print(n_loss)
    #print("state")
    #print(state)
    #print("Hin")
    #print(fed_init)
    batchlosses.append(n_loss)
    print("________________________________________________")
    sys.stdout.flush()
    
    if epoch!=prev_epoch:
        #sv_op.op.run(session=sess)
        
        saver = tf.train.Saver()
        os.system('mkdir save/epoch'+str(epoch))
        save_path = saver.save(sess, "save/epoch"+str(epoch)+"/model.ckpt")
        print("saved sess epoch")
        np.save("losses/batch"+str(batch_num), np.array(batchlosses))
        batchlosses=[]
        prev_epoch=epoch
    
    if batch_num%1000==0:
        #sv_op.op.run(session=sess)
        saver = tf.train.Saver()
        os.system('mkdir save/batchsave'+str(sb))
        save_path = saver.save(sess, "save/batchsave"+str(sb)+"/model.ckpt")
        print("saved sess batch")
        np.save("losses/batch"+str(batch_num), np.array(batchlosses))
        batchlosses=[]
        sb+=1





