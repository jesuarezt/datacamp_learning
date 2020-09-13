#Engineering features from strings - extraction

#The Length column in the hiking dataset is a column of strings, but contained in the column is the mileage for the hike. We're going to extract this mileage using regular expressions, and then use a lambda in Pandas to apply the extraction to the DataFrame.


# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())



#Engineering features from strings - tf/idf
#
#Let's transform the volunteer dataset's title column into a text vector, to use in a prediction task in the next exercise.

# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)



#Text classification using tf/idf vectors
#
#Now that we've encoded the volunteer dataset's title column into tf/idf vectors, let's use those vectors to try to predict the category_desc column.

#
    #Using train_test_split, split the text_tfidf vector, along with your y variable, into training and test sets. Set the stratify parameter equal to y, since the class distribution is uneven. Notice that we have to run the toarray() method on the tf/idf vector, in order to get in it the proper format for scikit-learn.
    #Use Naive Bayes' fit() method on the X_train and y_train variables.
    #Print out the score() of the X_test and y_test variables.



# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))