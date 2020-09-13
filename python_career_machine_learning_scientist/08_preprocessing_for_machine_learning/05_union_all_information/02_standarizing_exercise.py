#Extracting numbers from strings
#
#The length_of_time field in the UFO dataset is a text field that has the number of minutes within the string. Here, you'll extract that number from that text field using regular expressions.

def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")
    
    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda row:return_minutes(row) )

# Take a look at the head of both of the columns
print(ufo[['length_of_time','minutes']].head())


#Identifying features for standardization
#
#In this section, you'll investigate the variance of columns in the UFO dataset to determine which features should be standardized. After taking a look at the variances of the seconds and minutes column, you'll see that the variance of the seconds column is extremely high. Because seconds and minutes are related to each other (an issue we'll deal with when we select features for modeling), let's log normlize the seconds column.

# Check the variance of the seconds and minutes columns
print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo["seconds"])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())