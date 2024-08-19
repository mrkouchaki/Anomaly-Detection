# RNN Autoencoder for Anomaly Detection in 5G Network

## Overview

This project provides an RNN autoencoder for anomaly detection in time-series IQ samples from a 5G network. The primary goal is to detect jamming attacks by analyzing reconstruction errors from the autoencoder model. The code includes data loading, preprocessing, model training, and anomaly detection steps.

## Project Structure

- `data/`: Contains the data files used for training and testing.
- `src/`: Contains the source code for data generation, model training, anomaly detection, and plotting.
- `jammer/`: Contains the code for running the jammer.
- `README.md`: Provides an overview and instructions for the project.
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/software-package.git
    cd software-package
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Running the Jammer:**
    - Navigate to the `jammer` directory:
    ```sh
    cd jammer
    ```

    - Ensure you have UHD installed and an SDR (e.g., USRP) connected and powered on. To install UHD on Ubuntu, use the following commands:

    ```sh
    # Install dependencies
    sudo apt-get update
    sudo apt-get install libboost-all-dev libusb-1.0-0-dev python3-mako python3-numpy python3-requests cmake g++ libgmp-dev swig doxygen graphviz python3-scipy

    # Add Ettus Research PPA and install UHD
    sudo add-apt-repository ppa:ettusresearch/uhd
    sudo apt-get update
    sudo apt-get install libuhd-dev uhd-host

    # Verify the installation
    uhd_find_devices
    ```

    - Configure the `config.yaml` file to set up the jamming type, frequency, power, etc. Example `config.yaml`:
        ```yaml
        ---
         # The options below are applicable to all jammer types
         # Select Frequency operating range (1=2.4GHz, 2=5GHz) | default = 1
         band: 1
         # Select Jammer Type (1=constant, 2=sweeping, 3=random channel hopping) | default = 1
         jammer: 1
         # Select Type of Jamming (1=proactive, 2=reactive) | default = 1
         jamming: 1
         # Select Jamming waveform (1=single tone, 2=swept sine, 3=gaussian noise) | default = 3
         waveform: 2
         # Enter Jammer transmit power in dBm (Min = -40dBm, Max = 13dBm) | default = 6dBm
         power: 10
         # Enter channel jamming duration in sec | default = 10s
         t_jamming: 5
         
         # The options below are optional depending on the jammer of choice
         # Enter total runtime duration in sec | default = 200s
         duration: 100 # This option doesn't apply to constant jammer
         # Enter distance between adjacent channels in MHz (Min = 1MHz, Max = 20MHz) | default = 20MHz
         ch_dist: 1 # This option doesn't apply to constant jammer and 5GHz band
         # Enter the frequency to Jam in MHz | default = 2462MHz
         freq: 3440.5 # This option only applies to constant jammer
         # Select channel allocation (1=UNII-1, 2=UNII-2a, 3=UNII-2c, 4=UNII-3)
         allocation: 1 # This option only applies to 5GHz band
        ```

    - Run the `jammer.py` script to start the jamming process:
    ```sh
    python jammer.py
    ```

2. **Generate, Save, and Prepare your data:**
    - After setting up the 5G network and starting the jammer based on the configured setup, set the frequency and other parameters based on your 5G network settings.
    - Navigate to the `data` directory:
    ```sh
    cd data
    ```

    - Run the `data_generate.py` script to generate and save the `.dat` file containing IQ samples:
    ```sh
    python data_generate.py
    ```

    - Save your `.dat` or `.csv` files containing IQ samples in the `data/` directory.
    - Ensure your `.dat` files are formatted as binary files containing 32-bit floating-point numbers.

3. **Training the Model:**
    - Run the `model.py` script to train the RNN autoencoder on normal IQ samples (without jamming attacks).
    ```sh
    python src/model.py
    ```

4. **Anomaly Detection:**
    - Use the trained model to detect anomalies in IQ samples that may contain jamming attacks by running the `anomaly_detection.py` script.
    ```sh
    python src/anomaly_detection.py
    ```

### Detailed Documentation

#### Data Loading and Preprocessing

- `load_data(filepath)`: This function loads data from a specified file path. It supports `.csv` and `.dat` file formats. For `.csv` files, it reads a specific column containing the IQ samples. For `.dat` files, it reads binary-encoded data.

- `count_lines(filepath)`: This function counts the number of lines in a file, which is useful for understanding the size of the dataset.

#### Data Generator

The `DataGenerator` class is designed to handle batch-wise data loading and preprocessing. It reads data from the file, processes it to extract real and imaginary parts, normalizes the data, and returns it in batches suitable for training the RNN autoencoder.

- `__init__(self, filepath, batch_size, sequence_length, max_samples=None, for_training=True)`: Initializes the data generator with the file path, batch size, sequence length, maximum samples to process, and a flag indicating if the generator is used for training.

- `reset(self)`: Resets the generator state, reinitializing the file pointer and clearing internal buffers.

- `__iter__(self)`: Initializes the iterator, setting the file pointer to the beginning.

- `close(self)`: Closes the file handle to release resources.

- `process_data(self, samples)`: Processes the samples to extract and normalize the real and imaginary parts, and structures them into sequences.

- `__next__(self)`: Fetches the next batch of data, processes it, and returns it in a format suitable for the RNN autoencoder.

#### Model Definition and Training

The model is defined using `Sequential` from the `keras` library. It is an RNN autoencoder consisting of LSTM layers, a repeat vector layer, and time-distributed dense layers.

- `Sequential()`: Initializes a sequential model.
- `LSTM()`: Adds LSTM layers to the model.
- `RepeatVector()`: Repeats the input to match the sequence length.
- `TimeDistributed()`: Applies a dense layer to each time step of the sequence.

The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

Training the model involves:
- Initializing the `DataGenerator` with normal IQ samples.
- Iterating over epochs and steps to train the model on batches of data.
- Using `train_on_batch` to update the model weights for each batch.

#### Anomaly Detection and Plotting

Anomaly detection is performed using the trained model to predict IQ samples and calculate reconstruction errors.

- Generate predictions using the trained model.
- Calculate reconstruction errors for each sample.
- Determine the threshold for detecting anomalies based on the 99th percentile of the errors.
- Flag sequences with errors above the threshold as intrusions.
- Plot the original vs. reconstructed IQ samples for sequences where intrusions are detected, highlighting the detected intrusions.

### Testing

A test script is provided to validate the data generator and model setup using synthetic data. The script generates random complex numbers, saves them to a `.dat` file, and uses the data generator to load and process the data.

To run the test script:
```sh
python src/test_script.py
