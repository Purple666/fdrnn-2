import matplotlib.pyplot as plt
import numpy as np
data_num=10
losses=[]
for i in range(1,data_num):
    loss=np.load('losses1l/batch'+str(i)+'000.npy')
    #print(loss.shape)
    losses.append(np.mean(loss))
    print(loss)
    print(np.mean(loss))

plt.plot([i for i in range(1,data_num)], losses)
plt.xlabel('1000 batches')
plt.ylabel('loss')
plt.title('lambda=1')
plt.show()