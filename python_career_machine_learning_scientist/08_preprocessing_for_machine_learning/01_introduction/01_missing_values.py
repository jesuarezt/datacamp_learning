#Missing data - columns
#
#We have a dataset comprised of volunteer information from New York City. The dataset has a number of features, but we want to get rid of features that have at least 3 missing values.
#
#How many features are in the original dataset, and how many features are in the set after columns with at least 3 missing values are removed?
#
#    The dataset volunteer has been provided.
#    Use the dropna() function to remove columns.
#    You'll have to set both the axis= and thresh= parameters.

volunteer.dropna(axis =1, thresh=3).head()

#
#Missing data - rows
#
#Taking a look at the volunteer dataset again, we want to drop rows where the category_desc column values are missing. We're going to do this using boolean indexing, by checking to see if we have any null values, and then filtering the dataset so that we only have rows with those values.

# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isnull().sum())

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)