"""
Simple Implement of Product Quantization
"""
import time
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

SEED = 12
np.random.seed(SEED)


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.2f' % (func.__name__, time.time() - local_time))

    return wrapper


class PQ:
    def __init__(self, data, n_clusters=256, n_split=4, method_distance="euc"):
        self.data = data  # 2-d numpy array
        self.N = data.shape[0]
        self.D = data.shape[1]

        self.n_clusters = n_clusters  # num of cluster centers
        self.n_split = n_split  # num of sub-segments a vector is divided into
        assert D % n_split == 0

        self.method_distance = method_distance

        self.cluster_centers: np.array
        self.data_zipped: np.array

        self._init_with_data(self.data)

    @print_run_time
    def _init_with_data(self, data):
        """
        cluster and zip data
        """
        data_reshaped = data.reshape(-1, D // self.n_split)

        print("Calculate cluster center")
        km_model = KMeans(n_clusters=self.n_clusters)
        km_model.fit(data_reshaped)
        self.cluster_centers = km_model.cluster_centers_
        self.data_zipped = self.zip_data(data)
        print("Initialization finished.")

    def zip_data(self, data):
        """
        zip vectors in data, and represent them with indexes of cluster centers.
        """
        N, D = data.shape[0], data.shape[1]
        data_reshaped = data.reshape(N, self.n_split, -1)

        data_zipped = []
        for i in tqdm(range(N), desc="Zip data"):
            tmp = []
            for j in range(self.n_split):
                clus_idx2dist = {}
                for k in range(self.n_clusters):
                    dist = self.get_distance(data_reshaped[i][j], self.cluster_centers[k])
                    clus_idx2dist[k] = dist
                clus_idx2dist_sorted = sorted(clus_idx2dist.items(), key=lambda x: x[1])
                tmp.append(clus_idx2dist_sorted[0][0])  # most similar cluster center's idx
            data_zipped.append(tmp)
        return np.array(data_zipped)

    def get_distance(self, vec0, vec1):
        vec0 = vec0.reshape(1, -1)
        vec1 = vec1.reshape(1, -1)
        if self.method_distance == "euc":
            dist = euclidean_distances(vec0, vec1)
        elif self.method_distance == "cos":
            dist = cosine_distances(vec0, vec1)
        else:
            raise NotImplementedError
        return dist.sum()

    @print_run_time
    def get_most_similar(self, vec,
                         precise_mode=False, topn=10):
        """
        search by product quantization, O(n_clusters*n_split)
        """
        D = vec.shape[1]
        assert D == self.D

        min_dist = float('inf')
        ans_idx = -1
        idx2dist = {}

        vec_reshaped = vec.reshape(self.n_split, -1)

        # Create distance look up table.
        # No matter how big N is, the time complexity is fixed to O(n_clusters*n_split)
        dist_lookup = {}
        for i in range(self.n_clusters):
            dist_lookup[i] = {}
            for j in range(self.n_split):
                dist_lookup[i][j] = self.get_distance(self.cluster_centers[i], vec_reshaped[j])

        for i in range(self.N):
            total_dist = 0
            for j in range(self.n_split):
                total_dist += dist_lookup[self.data_zipped[i][j]][j]
            if total_dist < min_dist:
                ans_idx = i
                min_dist = total_dist
                idx2dist[i] = total_dist

        # When precise_mode is on, it will search topn nearest results by brute force.
        # Final result is close to the one searched by brute-force method, but still have bias :)
        if precise_mode:
            idx2dist_sorted = sorted(idx2dist.items(), key=lambda x: x[1])[:topn]
            idx_list = [item[0] for item in idx2dist_sorted]

            min_dist = float('inf')
            ans_idx = -1
            for idx in idx_list:
                dist = self.get_distance(vec, self.data[idx])
                if dist < min_dist:
                    ans_idx = idx
                    min_dist = dist

        print(ans_idx, min_dist)
        return ans_idx, min_dist

    @print_run_time
    def get_most_similar_brute_force(self, vec):
        """
        search by brute force method, O(N)
        """
        min_dist = float('inf')
        ans_idx = -1

        for i in range(self.N):
            dist = self.get_distance(vec, self.data[i])
            if dist < min_dist:
                ans_idx = i
                min_dist = dist

        print(ans_idx, min_dist)
        return ans_idx, min_dist


if __name__ == '__main__':
    N = 1000
    D = 128
    data = np.random.randn(N, D)
    pq = PQ(data, n_clusters=128, n_split=4)

    print("PQ vs. Brute force search")
    print("#1")
    rand_vec = np.random.randn(1, D)
    pq.get_most_similar(rand_vec, precise_mode=True, topn=20)
    print("#2")
    pq.get_most_similar_brute_force(rand_vec)
    print()
