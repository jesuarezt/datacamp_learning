from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)


############################################
# Comparing models

# Compare k nearest neighbors classifiers with k=1 and k=5 on the handwritten digits data set, which is already loaded into the variables X_train, y_train, X_test, and y_test. You can set k with the n_neighbors parameter when creating the KNeighborsClassifier object, which is also already imported into the environment.

# Which model has a higher test accuracy?

############################################
from sklearn.metrics import confusion_matrix

knn1 = KNeighborsClassifier(n_neighbors= 1)
knn1.fit(X_train, y_train)
pred = knn1.predict(X_test)

knn5 = KNeighborsClassifier(n_neighbors= 5)
knn5.fit(X_train, y_train)
pred2 = knn5.predict(X_test)

# calculate accuracy
print('1 neighbors')
print(confusion_matrix(y_test, pred))
print('5 neighbors')
print(confusion_matrix(y_test, pred2))

print('accuracy')
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
print(accuracy_score(y_test, pred2))








############################################