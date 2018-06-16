import numpy as np

from scipy.io import wavfile
import cmath

NUM_PARTS=38000
SAM_RATE=44100

def load_one_wav(i):
    fs, data = wavfile.read('parts/p'+str(i)+'.wav')
    ft_samp=np.fft.rfft(data)
    ind_ft_samp=[[abs(ft_samp[i]), cmath.phase(ft_samp[i]), i ] for i in range(len(ft_samp))]
    return ind_ft_samp


def make_y(x):
    f_polar=[xi[:2] for xi in x]
    f_future=(np.roll(f_polar, -1, axis=0)).tolist()
    return f_future


def batch_sequencer(raw_x, raw_y, batch_size, sequence_size, epoch_num):
    
    data_len = len(raw_x)
    dim_x=len(raw_x[0])
    dim_y=len(raw_y[0])
    print("dim_x")
    print(dim_x)
    print("dim_y")
    print(dim_y)
    batch_num = (data_len - 1) // (batch_size * sequence_size)
    rounded_data_len = batch_num * batch_size * sequence_size
    #xdata= np.array(raw_x)
    #ydata= np.array(raw_y)
    xdata=raw_x
    ydata=raw_y

    xdata = np.reshape(xdata[0:rounded_data_len], [batch_size, batch_num * sequence_size, dim_x])
    ydata = np.reshape(ydata[0:rounded_data_len], [batch_size, batch_num * sequence_size, dim_y])
    for epoch in range(epoch_num):
        for batch in range(batch_num):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            yield x, y, epoch, batch


def data_feeder( batch_size, sequence_size, epoch_num):
    real_input=np.load('real_input.npy')
    real_labels=np.load('real_labels.npy')

    print(real_input[2205*15:2205*15+100])
    print(real_labels[2205*15:2205*15+100])
    print(len(real_input))
    print(len(real_labels))
    #print(max(real_input))
    return batch_sequencer(real_input, real_labels , batch_size, sequence_size, epoch_num)

def save_data():
    real_input=[]
    real_labels=[]
    for i in range(0,NUM_PARTS):
        x=load_one_wav(i)
        real_input.append(x)
        real_labels.append(make_y(x))
    np.save('real_input.npy', np.reshape(np.array(real_input), [len(x)*NUM_PARTS, 3]))
    np.save('real_labels.npy', np.reshape(np.array(real_labels), [len(x)*NUM_PARTS, 2]))
    
if __name__ == "__main__":
    save_data()
    print("done, saved data")
    

