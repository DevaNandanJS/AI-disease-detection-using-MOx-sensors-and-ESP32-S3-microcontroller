import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import sys
import logging


# Configure logging at the top level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
                    
#Loading dataset
try:
    dataset= pd.read_csv(r"C:\ESP-32 ML Test\Dataset\dataset_preprocessed")
    logging.info("dataset loaded sucessfully")
except FileNotFoundError:
    logging.error("file not found in locatiion")
    sys.exit(1)
    
#seperating labels from dataset
x= dataset.drop(columns= 'Target')
y= dataset['Target']
logging.info("target removed and labels seperated scessfully")

#splitting the data for training and testing
x_train, x_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.2, 
    stratify= y, 
    shuffle= True, 
    random_state= 42)

logging.info(f"train shape; {x_train.shape}")
logging.info(f"test shape {x_test.shape}")

#addressing class imbalance
classes= np.unique(y_train)
sample_weight= compute_class_weight(class_weight= 'balanced',classes= classes, y= y_train)

#making weight dict
weight_dict= dict(zip(classes, sample_weight))
sample_weight_array= np.vectorize(weight_dict.get)(y_train)

#model
model= XGBClassifier(
    objective = "multi:softmax",
    num_class= len(classes),
    eval_metric= 'mlogloss',
    random_state= 42)

#training model
logging.info("training model")
model.fit(x_train, y_train, sample_weight= sample_weight_array)
logging.info("model trained sucessfully")

#evaluating the model
logging.info("testing the model")
predictions=model.predict(x_test)

#classififaction performance report
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))
