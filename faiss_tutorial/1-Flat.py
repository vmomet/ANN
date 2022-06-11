# 基于L2距离的暴力检索算法 by IndexFlatL2

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
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # 基于L2距离的暴力检索算法
print(index.is_trained)
index.add(xb)                  # 添加数据
print(index.ntotal)

k = 4                          # 距离最近的样本个数（k近邻个数）
# D, I = index.search(xb[:5], k) # sanity check（完整性检查?）
# print(I)
# print(D)
D, I = index.search(xq, k)     # 执行检索
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
