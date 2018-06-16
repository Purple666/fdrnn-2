import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import rnn 
import gen_wav
import random

dim_x=3
dim_y=2
N_LAYERS = 4
LAYER_SIZE=1024
BATCH_SIZE=1
chunk_size=(4410/2)+1
y=[0,0]
saved_states = tf.get_variable("saved_states", shape=[N_LAYERS,2,16, LAYER_SIZE])
with tf.Session() as sess:
  
  
    saver= tf.train.Saver()
    saver.restore(sess, "save/batchsave25/model.ckpt")
    print("saved_Hin")
    print(saved_states.eval())
    state=saved_states[:,:,:1,:].eval()
    state=np.zeros([N_LAYERS,2,BATCH_SIZE, LAYER_SIZE])
    saver = tf.train.import_meta_graph('save/batchsave25/model.ckpt.meta')
    saver.restore(sess, "save/batchsave25/model.ckpt")
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    # tf.get_default_graph().get_tensor_by_name("Y_predict:0")
    Y_predict =  tf.get_collection('Y_predict:0')
    p_keep= tf.get_default_graph().get_tensor_by_name("p_keep:0")
    Hin=tf.get_default_graph().get_tensor_by_name("Hin:0")
    batch_size=tf.get_default_graph().get_tensor_by_name("batch_size:0")
    seq_len=tf.get_default_graph().get_tensor_by_name("seq_len:0")
    H=tf.get_default_graph().get_tensor_by_name("H:0")
    Y_predict_amount=tf.get_default_graph().get_tensor_by_name("Y_predict_amount:0")
    Y_predict_angle=tf.get_default_graph().get_tensor_by_name("Y_predict_angle:0")
    for j in range(10):
        y=[random.random()*1000, random.random()]
        samp=[]
        for i in range(chunk_size*30):
            print("y")
            print(y)
            input=y[:]
            input.append(i%chunk_size)
            print("input")
            print(input)


            state, y, _, _, predict_amount, predict_angle =sess.run([H,Y_predict, batch_size, seq_len, Y_predict_amount, Y_predict_angle], feed_dict={X:np.reshape(input, [1,1,dim_x]), p_keep:1, Hin:state, batch_size:1, seq_len:1})
            #print("predict_amount")
            #print(predict_amount)
            #print("predict_angle")
            #print(predict_angle)
            y=[predict_amount[0][0],predict_angle[0][0]]

            samp.append(y)
        print(samp)
        #np.save('samp_fft',np.array(samp))
        print(len(samp))
        gen_wav.regen(samp, chunk_size, 'samples/gen'+str(j)+'.wav')


