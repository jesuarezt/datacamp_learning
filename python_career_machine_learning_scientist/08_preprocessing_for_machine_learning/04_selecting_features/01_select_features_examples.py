###Selecting relevant features
##
#Now let's identify the redundant columns in the volunteer dataset and perform feature selection on the dataset to return a DataFrame of the ##relevant features.
##
#For example, if you explore the volunteer dataset in the console, you'll see three features which are related to location: locality, ##region, and postalcode. They contain repeated information, so it would make sense to keep only one of the features.
##
#There are also features that have gone through the feature engineering process: columns like Education and Emergency Preparedness are a ##product of encoding the categorical variable category_desc, so category_desc itself is redundant now.
###
##Take a moment to examine the features of volunteer in the console, and try to identify the redundant features.

# Create a list of redundant column names to drop
to_drop = ["locality", "region", "category_desc", "created_date",'vol_requests']

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print(volunteer_subset.head())



#Checking for correlated features
#
#Let's take a look at the wine dataset again, which is made up of continuous, numerical features. Run Pearson's correlation coefficient on the dataset to determine which columns are good candidates for eliminating. Then, remove those columns from the DataFrame.

# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)