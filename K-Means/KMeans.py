import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----- Creating our own dataset! -----
np.random.seed(0) # --> Set a random seed
X, y = make_blobs(n_samples=4000, centers=[[4,4], [-2,-1], [2,-3], [1,1]], cluster_std=0.9)
plt.scatter(X[:,0], X[:,1], marker='.', alpha=0.3, ec='k', s=80) # --> X[:,0] = all x-values (first columm), X[:,1] = all y-values (second column)
plt.show()

# ----- Setting up K-Means -----
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
# --> init: initialization method of centroids, n_clusters: number of clusters after n_init and centroids to generate , n_init: number of times k-means will run with different centroid seeds.

# ----- Fitting the model -----
k_means.fit(X)
k_means_labels = k_means.labels_
print(k_means_labels)
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# ----- Creating the visual plot -----
fig = plt.figure(figsize=(6,4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels)))) # --> creates a unique color using plt's colormap for every unique cluster value in k_means_labels
ax = fig.add_subplot(1,1,1)

for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    # --> Create a list of all data points, where the data points that are 
    #in the cluster (ex. cluster 0) are labeled as true, else they are
    #labeled as false.
    my_members = (k_means_labels == k)
    # --> Define centroid
    cluster_center = k_means_cluster_centers[k]
    # --> Plots data point with color column
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor = col, marker = '.', ms = 10)
    # --> Plots the centroid with the specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = col, markeredgecolor = 'k', markersize = 6)

ax.set_title("K-Means")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

# ----- For 3 clusters -----
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)
k_means_3 = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=12)
k_means_3.fit(X)
k_means_3_labels = k_means_3.labels_
k_means_3_cluster_center = k_means_3.cluster_centers_
cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_3_labels))))

for k, col in zip(range(len(k_means_3_cluster_center)), cluster_colors):
    members = (k_means_3_labels == k)
    centroid = k_means_3_cluster_center[k]
    ax.plot(X[members, 0], X[members, 1], 'w', markerfacecolor=col, marker='.', ms=10)
    ax.plot(centroid[0], centroid[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title("K-Means (3 clusters)")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

#----- Customer Segmentation With KMeans -----
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'
df = pd.read_csv(path)
print(df.head(15))
df = df.drop('Address', axis = 1) # --> Drop categorical variables, KMeans does not really work on those type of variables.
df = df.dropna()
print(df.info())

# ----- Splitting data and Normalizing -----
X = df.values[:,1:] # --> Leaves out 'CustomerID' 
scaler = StandardScaler()
Clus_dataset = scaler.fit_transform(X)

# ----- Initializing KMeans Model -----
clusterNum = 3
k_means = KMeans(n_clusters = clusterNum, init = 'k-means++', n_init = 12, random_state = 42)
k_means.fit(Clus_dataset)
labels = k_means.labels_

# ----- Feature Engineering -----
df['Clus_km'] = labels # --> Assigning cluster labels to new 'Clus_km' column
cluster_summary = df.groupby('Clus_km').mean() # --> Groups by 'Clus_km' by computing mean of each feature input column
print(cluster_summary)
print(df.head())

# ----- Visualization: Based on age, income -----
area = np.pi * (Clus_dataset[:, 1])**2
plt.figure(figsize=(6, 4))
plt.scatter(Clus_dataset[:, 0], Clus_dataset[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k', alpha=0.5)
plt.xlabel('Age (scaled)', fontsize=18)
plt.ylabel('Income (scaled)', fontsize=16)
plt.tight_layout()
plt.savefig('customer_clusters.png', dpi=300)
print('Saved plot to customer_clusters.png')
plt.show()

# ----- Visualization in 3 dimensional space: Age, Income, Education -----
fig = px.scatter_3d(Clus_dataset, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))
fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))
fig.show()

# ----- Profile for each group, considering common characteristics of each cluster -----
df_sub = df[['Age', 'Edu', 'Income', 'Clus_km']].copy()
g = sns.pairplot(df_sub, hue='Clus_km', palette='viridis', diag_kind='kde')
g.fig.suptitle('Pairwise Scatter Plot with K-means Clusters', y=1.02)
g.fig.tight_layout()
g.fig.savefig('pairplot.png', dpi=300)
print('Saved pairplot to pairplot.png')
plt.show()

# --> The 3 clusters can be:
#- LATE CAREER, AFFLUENT, AND EDUCATED
#- MID CAREER AND MIDDLE INCOME
#- EARLY CAREER AND LOW INCOME