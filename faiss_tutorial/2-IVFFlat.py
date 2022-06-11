# 基于聚类检索算法 by IndexIVFFlat

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

# 预先将数据xb聚类，然后查询时先确定最近的簇，在簇周围基于FlatL2进行搜索

nlist = 100 # 聚类数量 
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

assert not index.is_trained
index.train(xb)                # IndexIVFFlat需要先train才能add
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search(nprobe=1)
print(I[-5:])                  # neighbors of the 5 last queries

index.nprobe = 10              # 待查询的簇数（默认1），有时邻居并不在同一个簇中，所以要多找几个。
                               # nprobe==nlist时，IndexIVFFlat等价于IndexFlatL2
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
