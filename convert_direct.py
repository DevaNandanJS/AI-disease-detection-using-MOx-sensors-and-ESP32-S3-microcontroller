import joblib
from hummingbird.ml import convert
import torch
import numpy as np
import pandas as pd
import ai_edge_torch

print("1. Loading the trained XGBoost model from joblib file")
clr = joblib.load("model.joblib")

print("2. Converting the XGBoost model to a PyTorch model via Hummingbird")
num_features = 6 
dummy_input = np.zeros((1, num_features), dtype=np.float32)
# Hummingbird converts the XGBoost model to a PyTorch module
hb_model = convert(clr, 'torch', dummy_input)

# The actual PyTorch module is inside hb_model.model
pt_model = hb_model.model
pt_model.eval()

# Convert the dummy input to a PyTorch tensor
sample_input = torch.from_numpy(dummy_input)

print("3. Converting the PyTorch model to TensorFlow Lite (Float32) via AI Edge Torch")
# Add a wrapper to ensure output matches expected format if necessary
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

wrapped_model = Wrapper(pt_model)

# Convert using ai_edge_torch
edge_model = ai_edge_torch.convert(wrapped_model, (sample_input,))
edge_model.export("model.tflite")
print("TensorFlow Lite (Float32) model saved to model.tflite")

print("4. Converting the PyTorch model to TensorFlow Lite (INT8) via AI Edge Torch + PTQ")
# Load the representative dataset for quantization
print("Loading representative dataset for calibration...")
dataset = pd.read_csv(r"C:\ESP-32 ML Test\Dataset\dataset_preprocessed")
X_quant = dataset.drop(columns='Target').values.astype(np.float32)

# Create a PTQ configuration using the AI Edge Torch API
def calibration_data_gen():
    # Provide ~100 samples for calibration
    for i in range(100):
        yield (torch.from_numpy(X_quant[i].reshape(1, num_features)),)

ptq_config = ai_edge_torch.quantize.PTQDefaultQuantizer(
    calibration_dataset=calibration_data_gen()
)

# Convert using ai_edge_torch with quantization
edge_model_quant = ai_edge_torch.convert(wrapped_model, (sample_input,), quant_config=ptq_config)
edge_model_quant.export("model_quant.tflite")
print("TensorFlow Lite (INT8) model saved to model_quant.tflite")

print("Conversion and Quantization complete.")
