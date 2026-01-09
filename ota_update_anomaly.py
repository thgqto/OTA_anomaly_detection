import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

# Fix random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Global variable to store X_train_normal for threshold calculation in detect_anomalies
X_train_normal_global = None

def simulate_ota_data(num_normal_updates=500, num_anomalous_updates=50, update_sequence_length=20):
    """
    Simulates sequences of OTA update process data, including normal and anomalous scenarios.
    Each update process is represented as a sequence of steps, where each step has several features.

    Normal updates follow a typical sequence of stages with expected resource usage and durations.
    Anomalies include unexpected resource spikes, stalled progress, or errors during critical stages.

    Args:
        num_normal_updates (int): Number of normal OTA update sequences to generate.
        num_anomalous_updates (int): Number of anomalous OTA update sequences to inject.
        update_sequence_length (int): The number of monitoring steps/snapshots in one full OTA update process.

    Returns:
        tuple: A tuple containing:
            - np.array: Simulated OTA update data with features for each step.
            - np.array: Labels for the simulated data (0 for normal, 1 for anomaly) for each update sequence.
    """
    print("Simulating OTA update data...")

    # Define update stages and typical behavior
    # Features: stage, cpu_usage, memory_usage, network_rx, network_tx, status_code, duration_sec
    # Stage Mapping: 0:Idle, 1:Download, 2:Verification, 3:Preparation, 4:Installation, 5:Finalization, 6:Completed
    normal_stage_profiles = {
        0: {'cpu': (0.01, 0.05), 'mem': (0.1, 0.2), 'net_rx': (0, 0), 'net_tx': (0, 0), 'status': 0, 'duration': (1, 5)},
        1: {'cpu': (0.2, 0.4), 'mem': (0.3, 0.5), 'net_rx': (50, 200), 'net_tx': (5, 20), 'status': 0, 'duration': (5, 15)},
        2: {'cpu': (0.3, 0.6), 'mem': (0.4, 0.6), 'net_rx': (10, 50), 'net_tx': (1, 10), 'status': 0, 'duration': (3, 10)},
        3: {'cpu': (0.4, 0.7), 'mem': (0.5, 0.7), 'net_rx': (0, 10), 'net_tx': (0, 5), 'status': 0, 'duration': (5, 12)},
        4: {'cpu': (0.5, 0.8), 'mem': (0.6, 0.8), 'net_rx': (0, 5), 'net_tx': (0, 5), 'status': 0, 'duration': (10, 25)},
        5: {'cpu': (0.1, 0.3), 'mem': (0.2, 0.4), 'net_rx': (0, 0), 'net_tx': (0, 0), 'status': 0, 'duration': (2, 8)},
        6: {'cpu': (0.01, 0.05), 'mem': (0.1, 0.2), 'net_rx': (0, 0), 'net_tx': (0, 0), 'status': 0, 'duration': (1, 5)}
    }

    all_update_sequences = []
    all_update_labels = []

    # Generate normal update sequences
    for _ in range(num_normal_updates):
        sequence = []
        current_stage = 0 # Start with Idle
        for i in range(update_sequence_length):
            if current_stage < 6 and np.random.rand() < 0.2: # Progress to next stage occasionally
                current_stage += 1
            
            profile = normal_stage_profiles[current_stage]
            step_data = [
                current_stage,
                np.random.uniform(*profile['cpu']),
                np.random.uniform(*profile['mem']),
                np.random.uniform(*profile['net_rx']),
                np.random.uniform(*profile['net_tx']),
                profile['status'],
                np.random.uniform(*profile['duration'])
            ]
            sequence.append(step_data)
        all_update_sequences.append(sequence)
        all_update_labels.append(0) # 0 for normal

    # Inject anomalous update sequences
    for _ in range(num_anomalous_updates):
        sequence = []
        current_stage = 0
        anomaly_injected = False
        anomaly_type = np.random.choice(['resource_spike', 'stalled_download', 'verification_error', 'unexpected_stage_jump', 'long_duration'])

        for i in range(update_sequence_length):
            # Normal progression for most steps
            if current_stage < 6 and np.random.rand() < 0.2:
                current_stage += 1
            
            profile = normal_stage_profiles[current_stage]
            step_data = [
                current_stage,
                np.random.uniform(*profile['cpu']),
                np.random.uniform(*profile['mem']),
                np.random.uniform(*profile['net_rx']),
                np.random.uniform(*profile['net_tx']),
                profile['status'],
                np.random.uniform(*profile['duration'])
            ]

            # Inject anomaly at a random step
            if not anomaly_injected and np.random.rand() < 0.3 and i > 5 and i < update_sequence_length - 5:
                if anomaly_type == 'resource_spike':
                    # High CPU/Memory during a low-activity stage (e.g., Idle or Finalization)
                    if current_stage in [0, 5, 6]:
                        step_data[1] = np.random.uniform(0.8, 1.0) # CPU
                        step_data[2] = np.random.uniform(0.8, 1.0) # Memory
                        anomaly_injected = True
                elif anomaly_type == 'stalled_download':
                    # Network Rx drops to near zero during Download stage
                    if current_stage == 1:
                        step_data[3] = np.random.uniform(0, 1) # Network RX
                        anomaly_injected = True
                elif anomaly_type == 'verification_error':
                    # Specific error code during Verification stage
                    if current_stage == 2:
                        step_data[5] = np.random.choice([101, 102, 103]) # Simulate an error code
                        anomaly_injected = True
                elif anomaly_type == 'unexpected_stage_jump':
                    # Jump to an out-of-sequence stage
                    if current_stage == 1: # From download, jump to installation
                        step_data[0] = 4
                        current_stage = 4
                        anomaly_injected = True
                elif anomaly_type == 'long_duration':
                    # Significantly longer duration for a step
                    step_data[6] = np.random.uniform(profile['duration'][1] * 2, profile['duration'][1] * 4)
                    anomaly_injected = True
            
            sequence.append(step_data)
        all_update_sequences.append(sequence)
        all_update_labels.append(1) # 1 for anomaly

    # Convert to numpy arrays
    X_data = np.array(all_update_sequences)
    y_labels = np.array(all_update_labels)

    # Shuffle the datasets
    p = np.random.permutation(len(X_data))
    X_data = X_data[p]
    y_labels = y_labels[p]

    return X_data, y_labels


def preprocess_ota_data(ota_data_sequences, scaler=None, fit_scaler=True):
    """
    Preprocesses the OTA update sequences for the Autoencoder.
    This involves flattening the sequences temporarily for scaling, then scaling features,
    and finally reshaping back into 3D sequences.

    Args:
        ota_data_sequences (np.array): Raw OTA data sequences (e.g., from simulate_ota_data).
                                      Shape: (num_updates, update_sequence_length, num_features).
        scaler (MinMaxScaler, optional): An existing scaler object to use for transformation.
                                         If None, a new scaler is created.
        fit_scaler (bool): If True, fits the scaler on the provided `ota_data_sequences`.
                           Should be True for training data, False for test/inference data.

    Returns:
        tuple: A tuple containing:
            - np.array: Scaled OTA data sequences ready for the Autoencoder.
            - MinMaxScaler: The fitted scaler object (if fit_scaler=True) or the provided scaler.
    """
    print("Preprocessing OTA update data (scaling and reshaping)...")
    
    num_updates, sequence_length, num_features = ota_data_sequences.shape
    
    # Flatten the sequences for scaling
    flat_data = ota_data_sequences.reshape(-1, num_features)

    if scaler is None:
        scaler = MinMaxScaler()
    
    if fit_scaler:
        scaled_flat_data = scaler.fit_transform(flat_data)
    else:
        scaled_flat_data = scaler.transform(flat_data)

    # Reshape back to sequences
    scaled_ota_data_sequences = scaled_flat_data.reshape(num_updates, sequence_length, num_features)
    
    return scaled_ota_data_sequences, scaler

def build_autoencoder_model(input_shape):
    """
    Builds an LSTM-based Autoencoder model tailored for sequences.

    The encoder compresses the input sequence into a latent space representation.
    The decoder reconstructs the input sequence from this latent space.
    The goal is to learn a compressed representation of "normal" sequential data.

    Args:
        input_shape (tuple): Shape of a single input sequence (update_sequence_length, num_features).

    Returns:
        tensorflow.keras.Model: The compiled Autoencoder model.
    """
    print("Building LSTM Autoencoder model...")
    timesteps, n_features = input_shape

    # Encoder
    encoder_inputs = Input(shape=(timesteps, n_features))
    encoder = LSTM(128, activation='relu', return_sequences=True)(encoder_inputs)
    encoder = LSTM(64, activation='relu', return_sequences=False)(encoder)
    latent_vector = Dense(32, activation='relu')(encoder) # Latent space representation

    # Decoder
    decoder = RepeatVector(timesteps)(latent_vector)
    decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
    decoder = LSTM(128, activation='relu', return_sequences=True)(decoder)
    decoder_outputs = TimeDistributed(Dense(n_features))(decoder)

    # Autoencoder model
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mse') # Mean Squared Error is common for autoencoders
    
    print(model.summary())
    return model

def train_autoencoder(model, X_train_normal, epochs=50, batch_size=32):
    """
    Trains the Autoencoder model on normal OTA update sequences.

    The model learns to reconstruct normal update process sequences. Anomalies, when presented
    to the trained model, will result in higher reconstruction errors.

    Args:
        model (tensorflow.keras.Model): The Autoencoder model to train.
        X_train_normal (np.array): Training data containing only normal OTA update sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: A tuple containing:
            - tensorflow.keras.Model: The trained Autoencoder model.
            - tensorflow.keras.callbacks.History: Training history.
    """
    print("Training Autoencoder model...")
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

    history = model.fit(X_train_normal, X_train_normal,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1, # Use a portion of training data for validation
                        callbacks=[early_stopping],
                        verbose=1)
    return model, history

def detect_anomalies(model, X_data_to_evaluate, threshold_percentile=95):
    """
    Detects anomalies in the input OTA update sequences using the trained Autoencoder.

    Anomalies are identified by high reconstruction errors (MSE) per sequence, compared to a
    threshold derived from the reconstruction errors of the training data (normal data).

    Args:
        model (tensorflow.keras.Model): The trained Autoencoder model.
        X_data_to_evaluate (np.array): OTA update sequences to check for anomalies.
        threshold_percentile (int): Percentile of reconstruction errors from the *normal training data*
                                     to use as the anomaly threshold.

    Returns:
        tuple: A tuple containing:
            - np.array: Boolean array indicating detected anomalies (True for anomaly).
            - np.array: Reconstruction errors for each sequence in X_data_to_evaluate.
            - float: The calculated anomaly threshold.
            - int: The threshold_percentile used. # Added this to return the percentile
    """
    print("Detecting anomalies...")
    # Get reconstruction errors for the global training data (normal only) to set a threshold
    global X_train_normal_global
    if X_train_normal_global is None:
        raise ValueError("X_train_normal_global must be set before calling detect_anomalies for threshold calculation.")

    X_train_pred = model.predict(X_train_normal_global, verbose=0)
    # Calculate MSE for each sequence across all timesteps and features
    train_reconstruction_errors = np.mean(np.square(X_train_normal_global - X_train_pred), axis=(1, 2))

    # Determine threshold based on a percentile of normal reconstruction errors
    # Anomalies should have errors significantly higher than normal data
    threshold = np.percentile(train_reconstruction_errors, threshold_percentile)
    print(f"Anomaly detection threshold set at {threshold_percentile}th percentile of training errors: {threshold:.4f}")

    # Predict and calculate reconstruction errors for the input data (X_data_to_evaluate)
    X_data_pred = model.predict(X_data_to_evaluate, verbose=0)
    data_reconstruction_errors = np.mean(np.square(X_data_to_evaluate - X_data_pred), axis=(1, 2))

    # Flag anomalies
    anomalies = data_reconstruction_errors > threshold
    return anomalies, data_reconstruction_errors, threshold, threshold_percentile # Returned threshold_percentile

def main():
    """
    Main function to run the OTA update anomaly detection prototype.
    Orchestrates data simulation, preprocessing, model building, training, and anomaly detection.
    """
    global X_train_normal_global # Access the global variable

    # --- 1. Simulate OTA Update Data ---
    update_sequence_length = 20 # Number of steps/snapshots in one OTA update process
    # Generate more normal data for training a robust model, fewer anomalies for testing
    raw_ota_data, ota_labels = simulate_ota_data(
        num_normal_updates=1000, 
        num_anomalous_updates=100, 
        update_sequence_length=update_sequence_length
    )
    
    # Separate normal data for training the autoencoder.
    # In a real scenario, this would be a large, verified dataset of normal OTA operations.
    ota_data_normal_only_for_training = raw_ota_data[ota_labels == 0]
    
    if len(ota_data_normal_only_for_training) == 0:
        print("Not enough normal data to train the model. Adjust simulation parameters.")
        return

    # --- 2. Preprocess Training Data ---
    # Fit the scaler ONLY on normal training data and store it
    X_train_normal_global, scaler = preprocess_ota_data(ota_data_normal_only_for_training, fit_scaler=True)
    
    # --- 3. Preprocess All Data for Evaluation ---
    # Apply the *already fitted* scaler to the entire dataset (normal + anomalies)
    X_all_updates_scaled, _ = preprocess_ota_data(raw_ota_data, scaler=scaler, fit_scaler=False)

    # --- 4. Build Autoencoder Model ---
    # input_shape is (sequence_length, num_features)
    input_shape = (X_train_normal_global.shape[1], X_train_normal_global.shape[2]) 
    model = build_autoencoder_model(input_shape)

    # --- 5. Train Autoencoder ---
    trained_model, history = train_autoencoder(model, X_train_normal_global, epochs=100, batch_size=64)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('OTA Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 6. Detect Anomalies ---
    # Define the percentile directly here or as a constant
    threshold_percentile_for_plot = 98 # <--- Defined here
    anomalies_detected, all_reconstruction_errors, threshold, _ = detect_anomalies(
        trained_model, X_all_updates_scaled, threshold_percentile=threshold_percentile_for_plot
    )

    # --- 7. Evaluation and Visualization ---
    print("\n--- Anomaly Detection Results for OTA Updates ---")
    print(f"Total OTA update sequences processed: {len(X_all_updates_scaled)}")
    print(f"Total actual anomalous OTA update sequences: {np.sum(ota_labels == 1)}")
    print(f"Total detected anomalous OTA update sequences: {np.sum(anomalies_detected)}")

    # Simple accuracy metrics
    true_positives = np.sum((anomalies_detected == True) & (ota_labels == 1))
    false_positives = np.sum((anomalies_detected == True) & (ota_labels == 0))
    true_negatives = np.sum((anomalies_detected == False) & (ota_labels == 0))
    false_negatives = np.sum((anomalies_detected == False) & (ota_labels == 1))

    print(f"True Positives (correctly identified anomalous updates): {true_positives}")
    print(f"False Positives (normal updates identified as anomalous): {false_positives}")
    print(f"True Negatives (correctly identified normal updates): {true_negatives}")
    print(f"False Negatives (anomalous updates missed): {false_negatives}")

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision:.4f}")
    else:
        print("Precision: N/A (no positive predictions)")
    
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall: {recall:.4f}")
    else:
        print("Recall: N/A (no actual anomalies)")

    # Plot reconstruction errors and threshold
    plt.figure(figsize=(15, 7))
    plt.plot(all_reconstruction_errors, marker='o', linestyle='', markersize=2, label='Reconstruction Error per Update')
    
    # Highlight actual anomalies in the plot
    actual_anomaly_indices = np.where(ota_labels == 1)[0]
    if len(actual_anomaly_indices) > 0:
        plt.plot(actual_anomaly_indices, all_reconstruction_errors[actual_anomaly_indices], 
                 'ro', markersize=4, label='Actual Anomalous Updates')
        
    # Use the captured threshold_percentile_for_plot here
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({threshold_percentile_for_plot}th percentile)')
    plt.title('OTA Update Anomaly Detection: Reconstruction Errors')
    plt.xlabel('OTA Update Sequence Index')
    plt.ylabel('Average Reconstruction Error (MSE) per Sequence')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Prototype Complete ---")
    print("This prototype demonstrates how an LSTM Autoencoder can be used for unsupervised anomaly detection on simulated OTA update processes.")
    print("In a real-world deployment, this model would monitor live telemetry streams from update agents and vehicle ECUs.")


if __name__ == "__main__":
    main()
