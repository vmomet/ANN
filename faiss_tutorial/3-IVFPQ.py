# 基于乘积量化（PQ）压缩的聚类检索算法 by IndexIVFPQ

import numpy as np

# 构建数据集
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# faiss
import faiss

# IndexIVFPQ在IndexIVFFlat基础上增加了PQ，用于压缩向量，减小存储和检索内存
# PQ是有损压缩，用子向量簇心id编码各段向量，因此在计算距离时并不是精确的

nlist = 100 # 聚类数量 
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8表明每个划分的子向量由8 bits编码（=256种）
index.train(xb)                   # IndexIVFPQ 需要先train才能add
index.add(xb)
# D, I = index.search(xb[:5], k)  # sanity check
# print(I)
# print(D)
index.nprobe = 10                 # 查询时，检索聚类中心数目
D, I = index.search(xq, k)        # search
print(I[-5:])
