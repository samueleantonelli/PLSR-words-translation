import numpy as np
from math import sqrt
from matplotlib import pyplot as plt, cm
from sklearn.decomposition import PCA


def readDM(dm_file):
    dm_dict = {}
    version = ""
    with open(dm_file) as f:
        dmlines=f.readlines()
    for l in dmlines:
        items=l.rstrip().split()
        row=items[0]
        vec=[float(i) for i in items[1:]]
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))


def neighbours(dm_dict,vec,n):
    cosines={}
    c=0
    for k,v in dm_dict.items():
        cos = cosine_similarity(vec, v)
        cosines[k]=cos
        c+=1
    c=0
    neighbours = []
    for t in sorted(cosines, key=cosines.get, reverse=True):
        if c<n:
             #print(t,cosines[t])
             neighbours.append(t)
             c+=1
        else:
            break
    return neighbours

def make_figure(m_2d, labels, savefile, figsize=(20, 20), xlim=(-1.5, 1.0), ylim=(-1.5, 1.0)):
    plt.figure(figsize=figsize)
    plt.scatter(m_2d[:, 0], m_2d[:, 1], c=[cm.rainbow(i) for i in range(len(labels))])
    for i, txt in enumerate(labels):
        plt.annotate(txt, (m_2d[i, 0], m_2d[i, 1]))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(savefile)


def run_PCA(dm_dict, savefile, figsize=(), xlim=(), ylim=()):
    m = []
    labels = []
    for k, v in dm_dict.items():
        labels.append(k)
        m.append(v)
    pca = PCA(n_components=3)
    pca.fit(m)
    m_2d = pca.transform(m)
    make_figure(m_2d, labels, savefile, figsize=figsize, xlim=xlim, ylim=ylim)

dm_dict_en = readDM('data/english.subset.dm')
dm_dict_cat = readDM('data/catalan.subset.dm')


