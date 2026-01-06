# CAN Bus Anomaly Detection using LSTM Autoencoder

## Overview

This repository presents a machine learning prototype for detecting anomalies on a Controller Area Network (CAN) bus, a crucial component in automotive systems. Given the increasing complexity of Software-Defined Vehicles (SDVs) and the critical role of Over-The-Air (OTA) updates, ensuring the integrity and security of in-vehicle communication is paramount. This prototype uses an LSTM (Long Short-Term Memory) Autoencoder, an unsupervised learning technique, to identify unusual patterns in simulated CAN bus traffic, which could indicate a cyberattack or a malfunctioning component.

## Why this is Critical

The focus is on securing the entire vehicle ecosystem, "from tail to nose." This includes protecting the communication networks within the vehicle itself, especially after OTA updates. Malicious actors could exploit vulnerabilities (potentially introduced or altered by an update) to inject forged messages, manipulate sensor data, or disrupt critical vehicle functions.

This prototype addresses the "nose" part of the security chain by demonstrating:

1.  **Proactive Threat Detection:** Identifying anomalies in real-time CAN bus data can provide early warnings of potential intrusions or system malfunctions.
2.  **Post-OTA Update Validation:** While OTA updates include robust integrity checks at the server side, behavioral monitoring within the vehicle post-update is essential. If an update inadvertently introduces a vulnerability, or if a compromised update somehow bypasses initial checks, an in-vehicle IDS like this can detect the resulting anomalous behavior.
3.  **Unsupervised Learning for Unknown Threats:** Traditional signature-based Intrusion Detection Systems (IDS) struggle with zero-day attacks. Autoencoders, by learning the "normal" behavior, can flag *any* significant deviation, making them effective against novel threats.
4.  **Foundation for Edge Computing:** This type of model can be deployed on in-vehicle edge computing units, enabling real-time anomaly detection without requiring continuous, high-bandwidth communication with cloud servers for every single CAN message.

## Technical Approach

The core of this anomaly detection system is an LSTM Autoencoder. Autoencoders are neural networks trained to reconstruct their input. When trained exclusively on "normal" data, they learn to efficiently encode and decode typical patterns. If presented with anomalous data, they will struggle to reconstruct it accurately, resulting in a high "reconstruction error." This error serves as the basis for anomaly detection.

### How it Works:

1.  **Data Simulation:** We generate synthetic CAN bus data, comprising normal messages (known IDs, typical data ranges) and various types of anomalies (unknown IDs, extreme data values, message bursts).
2.  **Data Preprocessing:**
    *   **Scaling:** Feature values (CAN ID, data bytes) are scaled to a common range (0-1) using `MinMaxScaler` to prevent features with larger magnitudes from dominating the learning process.
    *   **Sequencing:** CAN messages are grouped into sequences (sliding windows) to capture temporal dependencies, which is crucial for LSTM networks. This allows the model to understand the context of messages rather than just individual ones.
3.  **Model Building (LSTM Autoencoder):**
    *   **Encoder:** An LSTM network compresses the input sequence into a lower-dimensional "latent space" representation. This latent vector ideally captures the essential characteristics of the normal sequence.
    *   **Decoder:** Another LSTM network takes the latent vector and attempts to reconstruct the original input sequence.
4.  **Training:** The Autoencoder is trained *only on normal CAN bus data*. This forces the model to learn the patterns inherent in legitimate communication.
5.  **Anomaly Detection:**
    *   After training, the model reconstructs both normal and potentially anomalous data.
    *   The Mean Squared Error (MSE) between the original input and its reconstruction is calculated for each sequence.
    *   A high reconstruction error indicates a deviation from the learned normal patterns.
    *   An anomaly threshold is determined (e.g., the 98th percentile of reconstruction errors from the *training* data). Any sequence with a reconstruction error above this threshold is flagged as anomalous.

## Key Features

*   **Simulated CAN Bus Data:** Generates realistic-looking CAN data with configurable normal and anomalous messages (unknown IDs, extreme data, bursts).
*   **LSTM Autoencoder:** Implements a deep learning model capable of learning temporal patterns in CAN traffic.
*   **Unsupervised Learning:** Trains solely on normal data, eliminating the need for pre-labeled anomalous data.
*   **Feature Scaling:** Utilizes `MinMaxScaler` for robust data preparation.
*   **Sequence Generation:** Transforms raw message data into time-series sequences suitable for LSTMs.
*   **Dynamic Thresholding:** Calculates an anomaly threshold based on the distribution of reconstruction errors from normal training data.
*   **Performance Metrics:** Provides basic evaluation metrics (Precision, Recall) for the prototype's anomaly detection capabilities.
*   **Visualization:** Plots training loss and reconstruction errors, highlighting detected anomalies against actual anomalies for easy interpretation.

## How to Use

### Prerequisites

Before running the script, ensure you have Python 3.7+ installed along with the following libraries:

*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `tensorflow` (and `keras` which is included with TensorFlow 2.x)
*   `matplotlib`

You can install these using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### Running the Script

1.  **Clone the repository (or save the script):**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
    (Or simply save the provided Python code as `can_anomaly_detector.py`)

2.  **Execute the script:**
    ```bash
    python can_anomaly_detector.py
    ```

The script will:
*   Simulate CAN bus data.
*   Preprocess the data.
*   Build and train the LSTM Autoencoder model.
*   Display the model summary.
*   Plot the training and validation loss.
*   Perform anomaly detection on the simulated data.
*   Print evaluation metrics (True Positives, False Positives, Precision, Recall).
*   Generate a plot showing reconstruction errors over time, with the anomaly threshold and actual anomalies highlighted.

## Code Structure and Explanation

The script is organized into several functions, each responsible for a specific part of the anomaly detection pipeline:

*   `simulate_can_data(num_normal_messages, num_anomalies, sequence_length)`:
    *   **What it does:** Generates a synthetic dataset resembling CAN bus traffic. It includes normal messages with typical CAN IDs and data values, as well as injected anomalies.
    *   **Types of Anomalies:** `unknown_id` (using an ID not typically seen), `extreme_data` (values outside the normal operating range for a known ID), and `burst` (a rapid succession of messages for a single ID).
    *   **Returns:** Raw CAN data (features: CAN_ID, DATA_0, DATA_1) and corresponding ground-truth labels (0 for normal, 1 for anomaly) for evaluation purposes.

*   `create_sequences(data, sequence_length)`:
    *   **What it does:** Transforms the flat list of CAN messages into a 3D array of sequences (sliding windows). LSTMs process sequences, allowing them to learn temporal dependencies and context.
    *   **Why it's needed:** A single CAN message might not be anomalous on its own, but a sequence of messages (e.g., a rapid change in speed followed by a sudden brake command) could indicate an anomaly.

*   `preprocess_can_data(can_data, sequence_length, scaler=None, fit_scaler=True)`:
    *   **What it does:** Handles the scaling of numerical features and the creation of sequences.
    *   **Scaling:** Uses `MinMaxScaler` to normalize data between 0 and 1. It's crucial to `fit` the scaler only on the *training data* (normal data) and then `transform` all data (training and test) using the same fitted scaler to avoid data leakage.
    *   **Returns:** Scaled and sequenced data, along with the fitted `MinMaxScaler` object.

*   `build_autoencoder_model(input_shape)`:
    *   **What it does:** Constructs the LSTM Autoencoder neural network using TensorFlow/Keras.
    *   **Architecture:** Consists of an encoder (two LSTM layers) that reduces the input sequence to a dense latent representation, and a decoder (a `RepeatVector` layer followed by two LSTM layers and a `TimeDistributed(Dense)` layer) that attempts to reconstruct the original sequence from the latent representation.
    *   **Compilation:** Compiled with the `adam` optimizer and `mean_squared_error` loss, aiming to minimize the difference between input and output.

*   `train_autoencoder(model, X_train_normal, epochs, batch_size)`:
    *   **What it does:** Trains the Autoencoder model.
    *   **Key Principle:** It is trained *exclusively* on `X_train_normal` (sequences derived from normal CAN messages). This enables the model to learn the characteristics of legitimate traffic.
    *   **Early Stopping:** Includes an `EarlyStopping` callback to prevent overfitting and stop training once the validation loss stops improving.

*   `detect_anomalies(model, X_data, threshold_percentile)`:
    *   **What it does:** Uses the trained Autoencoder to identify anomalies in new or existing data.
    *   **Reconstruction Error:** For each input sequence in `X_data`, the model tries to reconstruct it. The MSE between the original and reconstructed sequence is the reconstruction error.
    *   **Thresholding:** A statistical threshold is calculated from the reconstruction errors of the *normal training data* (e.g., the 98th percentile). Any sequence in `X_data` with an error exceeding this threshold is flagged as an anomaly.
    *   **Returns:** A boolean array indicating anomalies, an array of reconstruction errors, and the calculated threshold.

*   `main()`:
    *   **What it does:** Orchestrates the entire process, calling the above functions in sequence.
    *   **Evaluation:** Calculates and prints common metrics like True Positives, False Positives, Precision, and Recall to give an idea of the prototype's performance.
    *   **Visualization:** Generates plots to visualize the training process and the anomaly detection results.

## Results & Interpretation

The script will output console messages indicating the progress and findings, followed by two plots:

1.  **Autoencoder Training Loss:** This plot shows the `loss` (MSE) during training and `val_loss` (validation loss) across epochs. A decreasing trend in both indicates that the model is learning effectively. The point where `val_loss` stabilizes or starts to increase might indicate where `EarlyStopping` would activate in a longer training run.

2.  **CAN Bus Anomaly Detection: Reconstruction Errors:**
    *   This plot displays the reconstruction error for every processed sequence.
    *   The **red dashed line** represents the dynamically calculated anomaly threshold. Data points above this line are classified as anomalous.
    *   **Red circles** indicate the sequences that were *actually* anomalous based on the simulation labels.
    *   Ideally, all red circles should appear above the dashed threshold line, and very few non-red points should be above it.

The printed evaluation metrics (`True Positives`, `False Positives`, `Precision`, `Recall`) offer a quantitative measure of the detector's effectiveness.

*   **True Positives (TP):** Anomalies correctly identified.
*   **False Positives (FP):** Normal sequences incorrectly flagged as anomalous.
*   **True Negatives (TN):** Normal sequences correctly identified as normal.
*   **False Negatives (FN):** Actual anomalies that were missed.
*   **Precision:** Out of all sequences flagged as anomalous, how many were actually anomalous (`TP / (TP + FP)`). A high precision means fewer false alarms.
*   **Recall:** Out of all actual anomalous sequences, how many were correctly detected (`TP / (TP + FN)`). A high recall means fewer missed anomalies.

Balancing precision and recall is critical. False positives can lead to unnecessary investigations and operational disruptions, while false negatives can mean missed attacks with potentially severe consequences.

## Further Improvements & Real-World Considerations

This prototype serves as a foundational demonstration. For real-world application, several enhancements and considerations are crucial:

*   **Real-World CAN Data:** Training and validating with actual CAN bus traces from various vehicle operating conditions (idle, driving, updates) is essential.
*   **Feature Engineering:** Exploring more sophisticated features beyond raw CAN ID and data bytes, such as message frequency, timing deviations, and sequence correlation.
*   **Hyperparameter Tuning:** Optimizing LSTM layers, latent dimension, sequence length, and training parameters for better performance.
*   **Threshold Adaptability:** Implementing dynamic threshold adjustment mechanisms rather than a static percentile, which could adapt to changing normal vehicle behavior over time.
*   **Edge Deployment Optimization:** Quantizing the model, optimizing for low-power inference, and integrating with real-time operating systems for in-vehicle deployment.
*   **Ensemble Methods:** Combining multiple anomaly detection models (e.g., statistical methods, other ML models) for more robust detection.
*   **Contextual Awareness:** Integrating vehicle state (e.g., driving, parked, charging, updating) into the anomaly detection logic.
*   **Scalability:** Designing the system to handle the high data rates of modern automotive Ethernet and multi-CAN networks.
*   **Explainability:** Developing methods to understand *why* a particular sequence was flagged as anomalous to aid in forensic analysis.

This prototype demonstrates a viable approach to enhancing automotive cybersecurity by leveraging AI/ML for in-vehicle anomaly detection, especially relevant in the context of securing OTA updates for Software-Defined Vehicles.

## Copyright Notice and Licensing
© 2026 Thiago Quinelato. All rights reserved.

This software and its associated documentation are provided for educational and research purposes only. The unauthorized use, reproduction, or distribution of this code, in whole or in part, for commercial purposes is strictly prohibited without explicit written permission from the copyright holder.
