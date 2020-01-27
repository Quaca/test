import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = -2 * np.random.rand(100,2)
Y = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = Y
plt.scatter(X[:,0],X[:,1], s = 50, x='b')
plt.show()


Kmeans = KMeans(n_clusters = 2)
Keans.fit(X)

Kmeans.cluster_centers_
#dobije koordinate

plt.scatter(X[:,0],X[:,1], s = 50, x='b')
plt.scatter(x,y, s = 200, x='g')
plt.scatter(x,y, s = 200, x='r')

Kmeans.labels_

sample_test = np.array([-3,3]).reshape(1,-1)
Kmeans.predict(sample_test)

