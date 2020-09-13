#Building a random forest model
##
#You'll again work on the Pima Indians dataset to predict whether an individual has diabetes. This time using a random forest classifier. #You'll fit the model on the training data after performing the train-test split and consult the feature importance values.
##
##The feature and target datasets have been pre-loaded for you as X and y. Same goes for the necessary packages and functions#.
# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))
# {'bmi': 0.09, 'insulin': 0.13, 'glucose': 0.21, 'diastolic': 0.08, 'triceps': 0.11, 'age': 0.16, 'pregnant': 0.09, 'family': 0.12}
#    77.6% accuracy on test set.

#Random forest for feature selection

#Now lets use the fitted random model to select the most important features from our input dataset X.

#The trained model from the previous exercise has been pre-loaded for you as rf.

# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Prints out the mask
print(mask)

# Apply the mask to the feature dataset X
reduced_X = X.loc[:,mask]

# prints out the selected column names
print(reduced_X.columns)
#     Index(['glucose', 'age'], dtype='object')

#Recursive Feature Elimination with random forests
#
#You'll wrap a Recursive Feature Eliminator around a random forest model to remove features step by step. This method is more conservative compared to selecting features after applying a single importance threshold. Since dropping one feature can influence the relative importances of the others.
#
#You'll need these pre-loaded datasets: X, X_train, y_train.
#
#Functions and classes that have been pre-loaded for you are: RandomForestClassifier(), RFE(), train_test_split().

# Wrap the feature eliminator around the random forest model
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask using an attribute of rfe
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)


### DELETE TWO FEATURES IN EACH STEP.
# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)