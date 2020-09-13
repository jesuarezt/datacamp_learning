#Identifying areas for feature engineering
##
#Take an exploratory look at the volunteer dataset, using the variable of that name. Which of the following columns would you want to #perform a feature engineering task on?#

volunteer[['vol_requests','title','created_date','category_desc']].head()
Out[5]: 
   vol_requests            ...                          category_desc
0            50            ...                                    NaN
1             2            ...              Strengthening Communities
2            20            ...              Strengthening Communities
3           500            ...              Strengthening Communities
4            15            ...                            Environment


#
#Encoding categorical variables - binary
#
#Take a look at the hiking dataset. There are several columns here that need encoding, one of which is the Accessible column, which needs to be encoded in order to be modeled. Accessible is a binary feature, so it has two values - either Y or N - so it needs to be encoded into 1s and 0s. Use scikit-learn's LabelEncoder method to do that transformation.

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking.Accessible)

# Compare the two columns
print(hiking[['Accessible_enc', 'Accessible']].head())


#Encoding categorical variables - one-hot
#
#One of the columns in the volunteer dataset, category_desc, gives category descriptions for the volunteer opportunities listed. Because it is a categorical variable with more than two categories, we need to use one-hot encoding to transform this column numerically. Use Pandas' get_dummies() function to do so.

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())