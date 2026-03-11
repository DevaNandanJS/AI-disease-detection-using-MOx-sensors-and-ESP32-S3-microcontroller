import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


dataset= pd.read_csv("Dataset/MOX-DISEASE_MERGED.csv")

#the threshold
threshold = 0.90

#calculating missing values for columns
missing_values= dataset.isnull().sum()
missing_percentage= missing_values/len(dataset)

cols_to_drop= missing_percentage[missing_percentage > threshold].keys()
dataset_processed= dataset.drop(columns= cols_to_drop) #drops all the columns  with null values > 90%
df= dataset_processed.drop(columns= 'Dataset ID')

#label encoding
le = LabelEncoder()
df['Target']= le.fit_transform(df['Target'])
print("datset  target labels have been encoded")
print(df['Target'].value_counts())


#saving datset after processing
df.to_csv('dataset_preprocessed', index= False)

print (f'shape of datset after dropping columns: {df.shape}')


