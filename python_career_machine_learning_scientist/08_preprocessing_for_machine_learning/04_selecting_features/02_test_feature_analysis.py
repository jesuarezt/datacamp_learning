##Exploring text vectors, part 1
##
#Let's expand on the text vector exploration method we just learned about, using the volunteer dataset's title tf/idf vectors. In this first part of text vector exploration, we're going to add to that function we learned about in the slides. We'll return a list of numbers with the function. In the next exercise, we'll write another function to collect the top words across all documents, extract them, and then use #that list to filter down our text_tfidf vector.

# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))



#Exploring text vectors, part 2
#
#Using the function we wrote in the previous exercise, we're going to extract the top words from each document in the text vector, return a list of the word indices, and use that list to filter the text vector down to those top words.

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]




#Training Naive Bayes with feature selection
#
#Let's re-run the Naive Bayes text classification model we ran at the end of chapter 3, with our selection choices from the previous exercise, on the volunteer dataset's title and category_desc columns.


# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(train_X, train_y)

# Print out the model's accuracy
print(nb.score(test_X, test_y))