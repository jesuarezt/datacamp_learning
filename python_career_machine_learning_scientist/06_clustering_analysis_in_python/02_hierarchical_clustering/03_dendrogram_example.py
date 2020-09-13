#Create a dendrogram
#
#Dendrograms are branching diagrams that show the merging of clusters as we move through the distance matrix. Let us use the Comic Con footfall data to create a dendrogram.
#
#The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time. cluster_labels has the cluster labels. A linkage object is stored in the variable distance_matrix.

# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendogram
plt.show()


#
#Timing run of hierarchical clustering
#
#In earlier exercises of this chapter, you have used the data of Comic-Con footfall to create clusters. In this exercise you will time how long it takes to run the algorithm on DataCamp's system.
#
#Remember that you can time the execution of small code snippets with:
#
#%timeit sum([1, 3, 2])
#
#The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time. The timeit module and linkage function are already imported
#
#How long does it take to the run the linkage function on the comic con data?



%timeit linkage(comic_con[['x_scaled', 'y_scaled']])




#FIFA 18: exploring defenders
#
#In the FIFA 18 dataset, various attributes of players are present. Two such attributes are:
#
#    sliding tackle: a number between 0-99 which signifies how accurate a player is able to perform sliding tackles
#    aggression: a number between 0-99 which signifies the commitment and will of a player
#
#These are typically high in defense-minded players. In this exercise, you will perform clustering based on these attributes in the data.
#
#This data consists of 5000 rows, and is considerably larger than earlier datasets. Running hierarchical clustering on this data can take up to 10 seconds.
#
#The following modules are pre-loaded: dendrogram, linkage, fcluster from scipy.cluster.hierarchy, matplotlib.pyplot as plt, seaborn as sns. The data is stored in a Pandas dataframe, fifa.

# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', hue='cluster_labels', data=fifa)
plt.show()