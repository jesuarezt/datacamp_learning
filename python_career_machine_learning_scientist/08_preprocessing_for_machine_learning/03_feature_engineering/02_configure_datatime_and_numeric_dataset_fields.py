#Engineering numerical features - taking an average
#
#A good use case for taking an aggregate statistic to create a new feature is to take the mean of columns. Here, you have a DataFrame of #running times named running_times_5k. For each name in the dataset, take the mean of their 5 run times.


# Create a list of the columns to average
run_columns = ['run1', 'run2', 'run3', 'run4', 'run5']

# Use apply to create a mean column
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1)

# Take a look at the results
print(running_times_5k)




#Engineering numerical features - datetime
#
#There are several columns in the volunteer dataset comprised of datetimes. Let's take a look at the start_date_date column and extract just the month to use as a feature for modeling.


# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
print(volunteer[['start_date_converted', 'start_date_month']].head())