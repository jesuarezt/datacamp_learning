#Manual feature extraction I
#
#You want to compare prices for specific products between stores. The features in the pre-loaded dataset sales_df are: storeID, product, quantity and revenue. The quantity and revenue features tell you how many items of a particular product were sold in a store and what the total revenue was. For the purpose of your analysis it's more interesting to know the average price per product.

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df.revenue/sales_df.quantity

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['quantity','revenue'], axis=1)

print(reduced_df.head())

#
#Manual feature extraction II
#
#You're working on a variant of the ANSUR dataset, height_df, where a person's height was measured 3 times. Add a feature with the mean height to the dataset, then drop the 3 original features.

# Calculate the mean height
height_df['height'] = height_df[['height_1','height_2','height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1','height_2','height_3'], axis=1)

print(reduced_df.head())


#
#Calculating Principal Components
#
#You'll visually inspect a 4 feature sample of the ANSUR dataset before and after PCA using Seaborn's pairplot(). This will allow you to inspect the pairwise correlations between the features.
#
#The data has been pre-loaded for you as ansur_df


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create a pairplot to inspect ansur_df
sns.pairplot(ansur_df)
plt.show()

# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(pc_df)
plt.show()



#
#PCA on a larger dataset
#
#You'll now apply PCA on a somewhat larger ANSUR datasample with 13 dimensions, once again pre-loaded as ansur_df. The fitted model will be used in the next exercise. Since we are not using the principal components themselves there is no need to transform the data, instead, it is sufficient to fit pca to the data.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)


#PCA explained variance
#
#You'll be inspecting the variance explained by the different principal components of the pca instance you created in the previous exercise.

# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())