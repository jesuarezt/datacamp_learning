#Exploring data types
#
#Taking another look at the dataset comprised of volunteer information from New York City, we want to know what types we'll be working with as we start to do more preprocessing.
#
#Which data types are present in the volunteer dataset?
#
#    The dataset volunteer has been provided.
#    Use the .dtypes attribute to check the datatypes.
volunteer.dtypes



#Converting a column type
#
#If you take a look at the volunteer dataset types, you'll see that the column hits is type object. But, if you actually look at the column, you'll see that it consists of integers. Let's convert that column to type int.


# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype('int')

# Look at the dtypes of the dataset
print(volunteer.dtypes)