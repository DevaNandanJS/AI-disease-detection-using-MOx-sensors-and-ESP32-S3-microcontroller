import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as  sns

try:
    dataset= pd.read_csv("Dataset/MOX-DISEASE_MERGED.csv")
    print ("dataset loaded sucessfully")
except FileNotFoundError:
    print("file not found")

print (f"the sample of the dataset is:\n{dataset.head()}")
print (f"the shape of the dataset is : {dataset.shape}")
print ("some basic infor on the dataset is:")
dataset.info()
print(f"the dataset descripttion is: {dataset.describe()}")


print(f"the dimentions of the dataset rows {dataset.shape[0]} and columns {dataset.shape[1]}")


#calculating missing values  
MissingValues= dataset.isnull().sum()
print(f"missing value count is: {MissingValues} and the percentage in thed datset is: {MissingValues/len(dataset)*100}")

#plotting
plt.figure(figsize=(12,6))
sns.heatmap(dataset.isnull(), cbar= False, cmap= 'viridis')
plt.title('missing values heatmap')
plt.show()

