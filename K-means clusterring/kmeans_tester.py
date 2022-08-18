import numpy as np
import random
import matplotlib.pyplot as plt

# Generating dataset
mean = [3,2]
cov = [[1, 0], [0, 1]]
def dataset(mean,cov):
    X,y = np.random.multivariate_normal(mean, cov, 200).T
    df = np.column_stack([X,y])
    return df

df1 = dataset(mean,cov)
mean = [1,5]
df2 = dataset(mean,cov)
mean = [0,1]
df3 = dataset(mean,cov)

data = np.concatenate((df1,df2,df3),axis=0)
plt.scatter(data[:,0], data[:,1],alpha=0.5, c = 'b')
plt.title("Input data")
plt.savefig("../results/Input_cluster.png")

plt.show()

# Class formation for k-means algorithm for applying it on N-dimensional data having 3 clusters
class KMeans:
    def __init__(self,n_clusters=3,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self,X):

        random_index = random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids = X[random_index]
        j=0

        for i in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X,cluster_group)
            j+=1
            # check finish
            if (old_centroids == self.centroids).all():
                
                print('Number of iterations needed: ',j)
                break
        return cluster_group

    def assign_clusters(self,X):
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
               
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self,X,cluster_group):
        new_centroids = []
        # To identify no. of clusters
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)


km = KMeans(n_clusters=3,max_iter=100)
y_means = km.fit_predict(data)

plt.scatter(data[y_means == 0,0],data[y_means == 0,1],alpha=0.5,color='red')
plt.scatter(data[y_means == 1,0],data[y_means == 1,1],alpha=0.5,color='blue')
plt.scatter(data[y_means == 2,0],data[y_means == 2,1],alpha=0.5,color='green')
plt.title("Clustered data")
plt.legend(["cluster 0", "cluster 1", "cluster 2"])
plt.savefig("../results/Output_cluster.png")
plt.show()


