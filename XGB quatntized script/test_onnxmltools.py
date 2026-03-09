import joblib
from onnxmltools import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnx2tf import convert as onnx2tf_convert
import tensorflow as tf
import numpy as np
import pandas as pd

print("1. Loading XGBoost model")
clr = joblib.load("model.joblib")

import traceback

try:
    print("2. Converting directly to ONNX using onnxmltools")
    initial_type = [('float_input', FloatTensorType([None, 6]))]
    onnx_model = convert_xgboost(clr, initial_types=initial_type)
    onnx_filename = "simple_model.onnx"
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX to {onnx_filename}")
except Exception as e:
    traceback.print_exc()
    import sys; sys.exit(1)
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Saved ONNX to {onnx_filename}")

print("3. Converting ONNX to TF")
tf_saved_model_dir = 'simple_tf_saved_model'
onnx2tf_convert(
    input_onnx_file_path=onnx_filename,
    output_folder_path=tf_saved_model_dir,
    non_verbose=True,
)

print("4. Converting TF to TFLite (Float32)")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open("simple_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved simple_model.tflite")
