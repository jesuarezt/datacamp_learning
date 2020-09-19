# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components = 6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)




### Review model information 

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index= titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway', :])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington', :])
