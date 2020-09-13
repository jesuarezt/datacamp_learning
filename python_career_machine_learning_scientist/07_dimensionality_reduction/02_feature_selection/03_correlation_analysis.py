#Inspecting the correlation matrix

#A sample of the ANSUR body measurements dataset has been pre-loaded as ansur_df. Use the terminal to create a correlation matrix for this dataset.

#What is the correlation coefficient between wrist and ankle circumference?

ansur_df.columns
#Index(['Elbow rest height', 'Wrist circumference', 'Ankle circumference', 'Buttock height', 'Crotch height'], dtype='object')

ansur_df[['Wrist circumference', 'Ankle circumference']].corr()


#Visualizing the correlation matrix
#
#Reading the correlation matrix of ansur_df in its raw, numeric format doesn't allow us to get a quick overview. Let's improve this by removing redundant values and visualizing the matrix using seaborn.
#
#Seaborn has been pre-loaded as sns, matplotlib.pyplot as plt, NumPy as np and pandas as pd.

# Create the correlation matrix
corr = ansur_df.corr()

# Draw the heatmap
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()


# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()


#Filtering out highly correlated features
#
#You're going to automate the removal of highly correlated features in the numeric ANSUR dataset. You'll calculate the correlation matrix and filter out columns that have a correlation coefficient of more than 0.95 or less than -0.95.
#
#Since each correlation coefficient occurs twice in the matrix (correlation of A to B equals correlation of B to A) you'll want to ignore half of the correlation matrix so that only one of the two correlated features is removed. Use a mask trick for this purpose.

# Calculate the correlation matrix and take the absolute value
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))



#Nuclear energy and pool drownings
#
#The dataset that has been pre-loaded for you as weird_df contains actual data provided by the US Centers for Disease Control & Prevention and Department of Energy.
#
#Let's see if we can find a pattern.
#
#Seaborn has been pre-loaded as sns and matplotlib.pyplot as plt.



# Print the first five lines of weird_df
print(weird_df.head(5))

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x='pool_drownings', y='nuclear_energy', data=weird_df)
plt.show()
# Print out the correlation matrix of weird_df
print(weird_df.corr())