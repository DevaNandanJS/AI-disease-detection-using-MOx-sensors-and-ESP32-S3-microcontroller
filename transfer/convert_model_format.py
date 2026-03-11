import joblib
import xgboost
import sklearn
import os

# Define the correct path to your original model file
MODEL_DIR = "model weghts"
MODEL_NAME = "model.joblib"
ORIGINAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# The output file will be created in the project root directory
OUTPUT_MODEL_PATH = "model.json"


print("--- XGBoost Model Format Converter ---")
print(f"  This script uses:")
print(f"  - xgboost version: {xgboost.__version__}")
print(f"  - scikit-learn version: {sklearn.__version__}")
print("-" * 40)

try:
    # Load the model from the old pickle file using the correct path
    print(f"Loading model from '{ORIGINAL_MODEL_PATH}'...")
    model = joblib.load(ORIGINAL_MODEL_PATH)
    print("Model loaded successfully.")

    # Save the model in XGBoost's modern, cross-compatible JSON format
    print(f"Re-saving model to '{OUTPUT_MODEL_PATH}'...")
    model.save_model(OUTPUT_MODEL_PATH)
    print("SUCCESS! Converted and saved model to 'model.json' in your project root.")
    print("You can now proceed with the main transpilation script.")

except FileNotFoundError:
    print(f"FATAL: Could not find model at '{ORIGINAL_MODEL_PATH}'.")
    print("Please ensure the path is correct and the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")
