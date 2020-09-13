#K-means clustering: first exercise
#
#This exercise will familiarize you with the usage of k-means clustering on a dataset. Let us use the Comic Con dataset and check how k-means clustering works on it.
#
#Recall the two steps of k-means clustering:
#
#    Define cluster centers through kmeans() function. It has two required arguments: observations and number of clusters.
#    Assign cluster labels through the vq() function. It has two required arguments: observations and cluster centers.
#
#The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time.

# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list =vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers, check_finite=True)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()



#Runtime of k-means clustering
#
#Recall that it took a significantly long time to run hierarchical clustering. How long does it take to run the kmeans() function on the FIFA dataset?
#
#The data is stored in a Pandas data frame, fifa. scaled_sliding_tackle and scaled_aggression are the relevant scaled columns. timeit and kmeans have been imported.
#
#Cluster centers are defined through the kmeans() function. It has two required arguments: observations and number of clusters. You can use %timeit before a piece of code to check how long it takes to run. You can time the kmeans() function for three clusters on the fifa dataset.




%timeit  kmeans(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 3)