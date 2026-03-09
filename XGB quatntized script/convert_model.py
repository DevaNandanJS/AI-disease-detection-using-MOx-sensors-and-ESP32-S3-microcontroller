import joblib
import onnx
from onnx2tf import convert as onnx2tf_convert
import tensorflow as tf
from hummingbird.ml import convert
import torch
import numpy as np
import pandas as pd
import os

print("1. Loading the trained XGBoost model from joblib file")
clr = joblib.load("model.joblib")

print("2. Converting the XGBoost model to a PyTorch model via Hummingbird")
# The preprocessed dataset has 6 float features based on the training script.
num_features = 6 
dummy_input = np.zeros((1, num_features), dtype=np.float32)
hb_model = convert(clr, 'torch', dummy_input)

print("3. Converting the PyTorch model to ONNX")
onnx_filename = 'model.onnx'
torch.onnx.export(
    hb_model.model,
    torch.from_numpy(dummy_input),
    onnx_filename,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
)
print(f"ONNX model saved to {onnx_filename}")

print("4. Loading the ONNX model")
onnx_model = onnx.load(onnx_filename)

print("5. Converting ONNX to TensorFlow SavedModel using onnx2tf")
tf_saved_model_dir = 'tf_saved_model'
onnx2tf_convert(
    input_onnx_file_path=onnx_filename,
    output_folder_path=tf_saved_model_dir,
    non_verbose=True,
)
print(f"TensorFlow SavedModel saved to {tf_saved_model_dir}")

print("6. Converting TensorFlow SavedModel to TensorFlow Lite (Float32)")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
# TFLite Micro requires static shapes.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

tflite_filename = 'model.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)
print(f"TensorFlow Lite (Float32) model saved to {tflite_filename}")

print("7. Converting TensorFlow SavedModel to TensorFlow Lite (INT8 Post-Training Quantization)")

# Load the representative dataset for quantization
print("Loading representative dataset for calibration...")
dataset = pd.read_csv(r"C:\ESP-32 ML Test\Dataset\dataset_preprocessed")
# Drop target to get only features
X_quant = dataset.drop(columns='Target').values.astype(np.float32)

def representative_data_gen():
    # Provide ~100 samples for the converter to calibrate min/max activations
    for i in range(100):
        # The TFLite model expects input shape [1, 6] based on the ONNX conversion
        yield [X_quant[i].reshape(1, num_features)]

# Re-initialize converter for INT8
converter_quant = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen

# Ensure that the model only uses integer operations
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8
converter_quant.inference_output_type = tf.int8

tflite_quant_model = converter_quant.convert()

tflite_quant_filename = 'model_quant.tflite'
with open(tflite_quant_filename, 'wb') as f:
    f.write(tflite_quant_model)
print(f"TensorFlow Lite (INT8) model saved to {tflite_quant_filename}")

print("Conversion and Quantization complete.")
