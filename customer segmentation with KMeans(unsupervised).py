from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X , _ = make_blobs(n_samples = 400 , centers = 4 , random_state = 42)
df = pd.DataFrame(X , columns = ["annual spending","spending score"])
print(df.shape)
print(df.head())
plt.scatter(df["annual spending"], df["spending score"])
kmeans = KMeans(n_clusters = 4 , random_state = 42)
kmeans.fit(X)
df["cluster"] = kmeans.labels_
print(df["cluster"].value_counts())
Inertia = []
for k in range(1,11):
    kmeans_model_test = KMeans(n_clusters = k , random_state = 42)
    kmeans_model_test.fit(X)
    Inertia.append(kmeans_model_test.inertia_)
plt.plot(range(1,11) ,Inertia , color = "black",marker = "o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
colors = ["red" , "cyan" , "magenta" , "purple"]
for cluster in range(4):
    cluster_data = df[df["cluster"] == cluster]
    plt.scatter(cluster_data["annual spending"] , cluster_data["spending score"] , color = colors[cluster] , label = f"group {cluster}")
plt.xlabel("annual spending")
plt.ylabel("spending score")
plt.scatter(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[:,1] , marker = "X" , color = "black" , s = 400 , label ="centroid")
plt.legend()
plt.show()