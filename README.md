# ESP32-S3 On-Device ML with XGBoost

This project demonstrates a complete end-to-end workflow for training a machine learning model (XGBoost) and deploying it for high-performance inference on an ESP32-S3 microcontroller.

The core idea is to transpile a trained model into optimized, dependency-free C code that can run directly on the device, allowing for fast, local, and private ML applications.

## Project Workflow

The project is divided into two main parts: a Python-based ML pipeline for training and conversion, and a C++ PlatformIO project for on-device execution and benchmarking.

### 1. ML Pipeline (Python)

The Python scripts manage the entire lifecycle of the model before it gets to the device.

1.  **`EDA.py` & `preprocessing.py`**:
    *   Loads the initial dataset (`.csv`).
    *   Performs Exploratory Data Analysis (EDA) and cleaning.
    *   Handles missing values and label-encodes the target variable.
    *   Saves a clean, processed dataset.

2.  **`split_dataset.py`**:
    *   Splits the preprocessed data into training and testing sets (`train.csv`, `test.csv`).

3.  **`train_model.py`**:
    *   Trains an `XGBClassifier` on the training data.
    *   Handles class imbalance to improve model performance.
    *   Saves the trained model as `model.joblib` and `model.bst`.

4.  **`convert_model_format.py`**:
    *   Converts the `model.joblib` pickle into `model.json`, a version-agnostic JSON format. This is crucial for ensuring compatibility with the transpiler.

5.  **`transpile_model.py`**:
    *   Uses the **`m2cgen`** (Model-to-Code-Generator) library to convert `model.json` into pure C code.
    *   Performs critical optimizations for the ESP32-S3:
        *   **Hardware Acceleration**: Replaces `double` with `float` to leverage the single-precision Floating Point Unit (FPU).
        *   **Memory Optimization**: Uses `PROGMEM` to store the model's constant arrays in flash memory, saving precious SRAM.
        *   **Speed Optimization**: Adds `IRAM_ATTR` to place the core `score()` function in high-speed IRAM.
    *   The final output is `src/xgb_model.h`, a self-contained C header file ready for the embedded application.

### 2. Embedded Application (PlatformIO & C++)

The PlatformIO project runs the transpiled model on the ESP32-S3.

*   **`platformio.ini`**: Configures the project for the ESP32-S3, enables PSRAM, and sets up a custom partition table with LittleFS for file storage.
*   **`data/` directory**: This directory holds the `test.csv` file that will be uploaded to the device's flash filesystem.
*   **`src/main.cpp`**:
    *   Initializes the device and LittleFS.
    *   Reads test data samples one by one from `/test.csv`.
    *   For each sample, it calls the `score(features)` function from `xgb_model.h` to run inference.
    *   Measures and benchmarks the performance of each inference, tracking latency and peak memory usage (SRAM and PSRAM).
    *   Prints a detailed performance report to the Serial Monitor.

## Hardware

*   **Board**: ESP32-S3-DevKitC-1 (or any similar ESP32-S3 board with PSRAM)
*   **Framework**: Arduino

## Software & Dependencies

### Python Environment

The Python scripts require the following libraries. Install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

*   `xgboost`
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `m2cgen`

### PlatformIO

*   **`platformio-core`**: The command-line utility for PlatformIO.
*   **Libraries**:
    *   `ArduinoJson`: For potential future extensions involving JSON.
    *   `LittleFS_esp32`: Filesystem support for reading the test data.

## How to Use This Project

### Step 1: Set up the Python Environment

Make sure you have Python installed, then install the required packages.

```bash
pip install -r requirements.txt
```

### Step 2: Run the Full ML Pipeline

Execute the Python scripts in the correct order to generate the C header file (`src/xgb_model.h`).

1.  **Preprocess the data**:
    ```bash
    python preprocessing.py
    ```
2.  **Split the dataset**:
    ```bash
    python split_dataset.py
    ```
3.  **Train the model**:
    ```bash
    python train_model.py
    ```
4.  **Convert the model to JSON**:
    ```bash
    python convert_model_format.py
    ```
5.  **Transpile the model to C code**:
    ```bash
    python transpile_model.py
    ```

After this, the `src/xgb_model.h` file will be created or updated with the newly trained model.

### Step 3: Set up the Hardware

Connect your ESP32-S3 board to your computer.

### Step 4: Build and Upload the Filesystem

The `test.csv` file needs to be uploaded to the device's flash memory using LittleFS.

```bash
pio run --target uploadfs
```

### Step 5: Build, Upload, and Monitor the Application

Now, compile and upload the main firmware.

```bash
pio run --target upload
```

Open the Serial Monitor to view the output. The benchmark will start when you send any character.

```bash
pio device monitor
```

You should see a report like this:

```
--- ESP32-S3 XGBoost Inference Benchmark ---
Send any character to start the benchmark.
Processing /test.csv...

--- Run Report ---
Total Inferences: 1000
Average Latency per Inference: 150 us
Peak Internal SRAM Usage: 512 bytes
Peak PSRAM Usage: 0 bytes

--- Benchmark Complete ---
Send any character to run again.
```

## Project Structure

```
.
├── data/
│   └── test.csv           # Test data for the device
├── Dataset/
│   ├── train.csv          # Training data for the model
│   └── test.csv           # Test data for model evaluation in Python
├── src/
│   ├── main.cpp           # Main application logic for the ESP32
│   └── xgb_model.h        # Transpiled C code of the XGBoost model
├── .gitignore
├── convert_model_format.py # Python script to convert model to JSON
├── EDA.py                 # Python script for exploratory data analysis
├── partitions.csv         # Custom partition table for the ESP32
├── platformio.ini         # PlatformIO project configuration
├── preprocessing.py       # Python script for data cleaning
├── requirements.txt       # Python dependencies
├── split_dataset.py       # Python script to split the dataset
├── train_model.py         # Python script to train the XGBoost model
└── transpile_model.py     # Python script to convert the model to C code
```
