import time
import scipy.io as scio
from sklearn import manifold
from tsnecuda import TSNE
import numpy as np
import matplotlib.pyplot as plt
import argparse
print(time.asctime(time.localtime(time.time())))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/s', help='path to s1~s6.mat')
    parser.add_argument('--session', default=[1,2,3,4,5,6], help='s1~s6')
    parser.add_argument('--sort', type=str, default='unsort', help='[unsort|sorted]')
    parser.add_argument('--tsne', type=str, default='sklearn_tsne', help='[sklearn_tsne|cuda_tsne]')
    parser.add_argument('--perplexity', default=30, help='perplexity 5~50')
    parser.add_argument('--lr', default=500, help='learning_rate 10~1000')
    return parser.parse_args()
parser=options()

##Data
s=0
for Session in parser.session:#Session1~6
    ##Firing rate(input)
    data = scio.loadmat(parser.dataroot+str(Session)+".mat")
    FR=np.zeros([192,data['bin_firing_rate'].shape[1],data['bin_firing_rate'][0][0].shape[1]])
    for i in range(FR.shape[0]):
        for j in range(FR.shape[1]):
            FR[i][j]=data['bin_firing_rate'][i][j]
    #FR shape([192, 3, 12777])
    ##sorted/unsort
    if parser.sort == 'unsort':
        FR=sum(FR.transpose((1,0,2)))
    else:
        FR=FR.reshape(-1,data['bin_firing_rate'][0][0].shape[1])
    ##label
    label=np.full([FR.shape[1]],Session)
    if(s!=0):
        fr_cs=np.concatenate((fr_cs,FR.transpose((1,0))))
        label_cs=np.concatenate((label_cs,label))
    else:
        fr_cs=FR.transpose((1,0))
        label_cs=label
        s+=1
    ##movement(label)
#     finger = torch.from_numpy(data['finger'])[1:].transpose(1,0).float()
#     finger = (finger[1:3].transpose(0,1)*(-10))
#     vel_f=torch.from_numpy(data['vel_f']).float()
#     vel_f=(vel_f[1:3].transpose(0,1)*(-10)/0.064)
#     acc_f = torch.from_numpy(data['acc_f']).float()
#     acc_f = (acc_f[1:3].transpose(0,1)*-10/0.004096)
#     acc_f = torch.cat((torch.zeros(1,2),acc_f),0)#補零=vel_f.size()
print('fr_cs',fr_cs.shape)
print('label_cs',label_cs.shape)

##t-SNE
if parser.tsne == 'tsne':
    tsne = manifold.TSNE(n_components=2,perplexity=parser.perplexity,learning_rate=parser.lr)
    fr_cs_tsne = tsne.fit_transform(fr_cs)
else:
    fr_cs_tsne = TSNE(n_components=2, perplexity=parser.perplexity, learning_rate=parser.lr).fit_transform(fr_cs)#perplexity困惑:5~50
print('fr_cs_tsne',fr_cs_tsne.shape)

##fig
plt.show()
plt.figure(figsize=(16,13))
plt.scatter(fr_cs_tsne[:,0],fr_cs_tsne[:,1],c=label_cs.astype(int),
            cmap='Paired',
            marker='.',
            alpha=0.6,)#透明度
plt.colorbar()
plt.savefig(time.asctime(time.localtime(time.time()))+'.png')
print(time.asctime(time.localtime(time.time())))

