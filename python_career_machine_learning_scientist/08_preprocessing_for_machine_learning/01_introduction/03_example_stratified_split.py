##Class imbalance
##
##In the volunteer dataset, we're thinking about trying to predict the category_desc variable using the other features in the dataset. First, ##though, we need to know what the class distribution (and imbalance) is for that label.
##
##Which descriptions occur less than 50 times in the volunteer dataset?
##
    ##The dataset volunteer has been provided.
    ##The colum you want to check is category_desc.
    ##Use the value_counts() method to check variable counts.


volunteer.category_desc.value_counts()

#Strengthening Communities    307
#Helping Neighbors in Need    119
#Education                     92
#Health                        52
#Environment                   32
#Emergency Preparedness        15
#Name: category_desc, dtype: int64


#
#Stratified sampling
#
#We know that the distribution of variables in the category_desc column in the volunteer dataset is uneven. If we wanted to train a model to try to predict category_desc, we would want to train the model on a sample of data that is representative of the entire dataset. Stratified sampling is a way to achieve this.

# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X,volunteer_y, stratify= volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())