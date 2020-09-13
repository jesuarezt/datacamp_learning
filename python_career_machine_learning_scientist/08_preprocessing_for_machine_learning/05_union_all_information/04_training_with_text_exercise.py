#Selecting the ideal dataset
#
#Let's get rid of some of the unnecessary features. Because we have an encoded country column, country_enc, keep it and drop other columns related to location: city, country, lat, long, state.
#
#We have columns related to month and year, so we don't need the date or recorded columns.
#
#We vectorized desc, so we don't need it anymore. For now we'll keep type.
#
#We'll keep seconds_log and drop seconds and minutes.
#
#Let's also get rid of the length_of_time column, which is unnecessary after extracting minutes.

# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds','seconds_log','minutes']].corr())

# Make a list of features to drop
to_drop = ['city', 'country', 'lat', 'long', 'state','date','recorded','desc','minutes','seconds','length_of_time']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)


#Modeling the UFO dataset, part 1
#
#In this exercise, we're going to build a k-nearest neighbor model to predict which country the UFO sighting took place in. Our X dataset has the log-normalized seconds column, the one-hot encoded type columns, as well as the month and year when the sighting took place. The y labels are the encoded country column, where 1 is us and 0 is ca.

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X,y,stratify=y)

# Fit knn to the training sets
knn.fit(train_X, train_y)

# Print the score of knn on the test sets
print(knn.score(test_X, test_y))



#Modeling the UFO dataset, part 2
#
#Finally, let's build a model using the text vector we created, desc_tfidf, using the filtered_words list to create a filtered text vector. Let's see if we can predict the type of the sighting based on the text. We'll use a Naive Bayes model for this.

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit nb to the training sets
nb.fit(train_X, train_y)

# Print the score of nb on the test sets
print(nb.score(test_X, test_y))