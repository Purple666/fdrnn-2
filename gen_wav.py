import numpy as np

from scipy.io import wavfile
from math import pi
from math import log
import cmath
PIECE_LEN=4410
CHUNK_LEN=PIECE_LEN//2 +1


def load_fft(f):
    fs, data = wavfile.read(f)
    ft_samp=np.fft.rfft(data[:PIECE_LEN])
    amounts_angles=[[abs(f),cmath.phase(f)] for f in ft_samp]
    return amounts_angles


def regen(fd, c_len, file_name):
    fd_len=len(fd)
    chunk_n=fd_len//c_len
    samp_regen=np.array([])
    for i  in range(chunk_n):
        chunk_fd=fd[i*c_len:(i+1)*c_len]
        regen_ft=[0.0001*fi[0]*cmath.exp(complex(0, fi[1])) for fi in chunk_fd]
        samp_regen=np.concatenate((samp_regen, np.fft.irfft(regen_ft)))
    wavfile.write(file_name, 44100, samp_regen)
    return samp_regen


#fd=np.empty([0,2])
#for i in range(20):
#    chunk_fd = load_fft('p'+str(i)+'.wav')
#    print(np.array(chunk_fd).shape)
#    print(fd.shape)
#    fd=np.concatenate((fd, chunk_fd))
#
#
#wavfile.write('regen.wav', 44100, regen(fd, CHUNK_LEN))