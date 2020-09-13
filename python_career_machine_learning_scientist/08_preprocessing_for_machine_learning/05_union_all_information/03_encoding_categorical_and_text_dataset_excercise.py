#Encoding categorical variables
#
#There are couple of columns in the UFO dataset that need to be encoded before they can be modeled through scikit-learn. You'll do that transformation here, using both binary and one-hot encoding methods.

# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda row: 1 if row == 'us' else 0)

# Print the number of unique type values
print(len(ufo["type"].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo["type"])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)



#Features from dates
#
#Another feature engineering task to perform is month and year extraction. Perform this task on the date column of the ufo dataset.

# Look at the first 5 rows of the date column
print(ufo["date"].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda x: x.month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda x: x.year)

# Take a look at the head of all three columns
print(ufo[['date','month','year']].head())



#Text vectorization
#
#Let's transform the desc column in the UFO dataset into tf/idf vectors, since there's likely something we can learn from this field.
# Take a look at the head of the desc field
print(ufo["desc"].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit(ufo["desc"])

# Look at the number of columns this creates
# Take a look at the head of the desc field
print(ufo["desc"].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo["desc"])

# Look at the number of columns this creates
print(desc_tfidf.shape)