import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def split_and_save_dataset():
    dataset_path = r"C:\ESP-32 ML Test\Dataset\dataset_preprocessed"
    output_dir = r"C:\ESP-32 ML Test\Dataset"
    
    # Loading dataset
    try:
        dataset = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully")
    except FileNotFoundError:
        logging.error(f"File not found at location: {dataset_path}")
        sys.exit(1)
        
    # Separating labels from dataset
    x = dataset.drop(columns='Target')
    y = dataset['Target']
    logging.info("Target removed and labels separated successfully")

    # Splitting the data for training and testing exactly as in train_model.py
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2, 
        stratify=y, 
        shuffle=True, 
        random_state=42
    )
    
    logging.info(f"Train features shape: {x_train.shape}")
    logging.info(f"Test features shape: {x_test.shape}")

    # Recombine features and labels
    train_dataset = x_train.copy()
    train_dataset['Target'] = y_train
    
    test_dataset = x_test.copy()
    test_dataset['Target'] = y_test

    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_dataset.to_csv(train_path, index=False)
    test_dataset.to_csv(test_path, index=False)
    
    logging.info(f"Train dataset saved to: {train_path}")
    logging.info(f"Test dataset saved to: {test_path}")

if __name__ == "__main__":
    split_and_save_dataset()
