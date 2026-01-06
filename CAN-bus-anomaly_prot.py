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

def simulate_can_data(num_normal_messages=10000, num_anomalies=100, sequence_length=10):
    """
    Simulates CAN bus data including normal operational patterns and injected anomalies.

    Normal data: Consists of a few common CAN IDs with data bytes changing slowly.
    Anomalies: Includes unknown CAN IDs, extreme data values, and high frequency bursts.

    Args:
        num_normal_messages (int): Number of normal CAN messages to generate.
        num_anomalies (int): Number of anomalous events to inject.
        sequence_length (int): The length of each message sequence (window) for the ML model.

    Returns:
        tuple: A tuple containing:
            - np.array: Simulated CAN data with features (CAN_ID, DATA_0, DATA_1).
            - np.array: Labels for the simulated data (0 for normal, 1 for anomaly).
    """
    print("Simulating CAN bus data...")
    # Define common CAN IDs and their typical data ranges
    normal_can_ids = [0x100, 0x101, 0x200]
    data_ranges = {
        0x100: {'DATA_0': (0, 100), 'DATA_1': (0, 50)}, # e.g., speed and RPM
        0x101: {'DATA_0': (20, 30), 'DATA_1': (5, 15)}, # e.g., temperature and pressure
        0x200: {'DATA_0': (0, 1), 'DATA_1': (0, 1)}    # e.g., boolean states
    }

    messages = []
    labels = []

    # Generate normal messages
    for i in range(num_normal_messages):
        can_id = np.random.choice(normal_can_ids)
        
        # Simulate slow changes for normal data
        data_0 = np.random.randint(data_ranges[can_id]['DATA_0'][0], data_ranges[can_id]['DATA_0'][1] + 1)
        data_1 = np.random.randint(data_ranges[can_id]['DATA_1'][0], data_ranges[can_id]['DATA_1'][1] + 1)
        
        messages.append([can_id, data_0, data_1])
        labels.append(0) # 0 for normal

    # Inject anomalies
    for i in range(num_anomalies):
        anomaly_type = np.random.choice(['unknown_id', 'extreme_data', 'burst'])

        if anomaly_type == 'unknown_id':
            # Inject an unknown CAN ID
            can_id = 0x7FF # A typical high-value, unused ID
            data_0 = np.random.randint(0, 256)
            data_1 = np.random.randint(0, 256)
            messages.append([can_id, data_0, data_1])
            labels.append(1)
        elif anomaly_type == 'extreme_data':
            # Inject extreme data values for a known CAN ID
            can_id = np.random.choice(normal_can_ids)
            # Use values far outside typical range
            data_0 = np.random.choice([data_ranges[can_id]['DATA_0'][0] - 100, data_ranges[can_id]['DATA_0'][1] + 100]) 
            data_1 = np.random.choice([data_ranges[can_id]['DATA_1'][0] - 50, data_ranges[can_id]['DATA_1'][1] + 50])
            messages.append([can_id, data_0, data_1])
            labels.append(1)
        elif anomaly_type == 'burst':
            # Inject a burst of messages for a single CAN ID
            can_id = np.random.choice(normal_can_ids)
            for _ in range(np.random.randint(5, 20)): # 5 to 20 messages in a burst
                data_0 = np.random.randint(data_ranges[can_id]['DATA_0'][0], data_ranges[can_id]['DATA_0'][1] + 1)
                data_1 = np.random.randint(data_ranges[can_id]['DATA_1'][0], data_ranges[can_id]['DATA_1'][1] + 1)
                messages.append([can_id, data_0, data_1])
                labels.append(1) # Label all burst messages as anomaly

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(messages, columns=['CAN_ID', 'DATA_0', 'DATA_1'])
    df['Label'] = labels

    # Shuffle the dataset to mix normal and anomalous data for training/testing purposes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Note: For an Autoencoder, we typically train only on normal data.
    # The 'labels' here are primarily for evaluating the detector later.
    return df[['CAN_ID', 'DATA_0', 'DATA_1']].values, df['Label'].values

def create_sequences(data, sequence_length):
    """
    Creates sequences (sliding windows) from the raw data.
    This prepares the data for an LSTM Autoencoder, which expects sequences as input.

    Args:
        data (np.array): The input data (e.g., preprocessed CAN messages).
        sequence_length (int): The desired length of each sequence.

    Returns:
        np.array: A 3D numpy array where each element is a sequence.
                  Shape: (num_sequences, sequence_length, num_features).
    """
    print(f"Creating sequences of length {sequence_length}...")
    sequences = []
    # Ensure there's enough data to form at least one sequence
    if len(data) < sequence_length:
        return np.array([]) # Return empty array if not enough data
        
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def preprocess_can_data(can_data, sequence_length, scaler=None, fit_scaler=True):
    """
    Preprocesses the CAN data for the Autoencoder.
    This involves scaling the features and creating sequences.

    Args:
        can_data (np.array): Raw CAN data (e.g., from simulate_can_data).
        sequence_length (int): The length of each message sequence (window).
        scaler (MinMaxScaler, optional): An existing scaler object to use for transformation.
                                         If None, a new scaler is created.
        fit_scaler (bool): If True, fits the scaler on the provided `can_data`.
                           Should be True for training data, False for test/inference data.

    Returns:
        tuple: A tuple containing:
            - np.array: Scaled and sequenced data ready for the Autoencoder.
            - MinMaxScaler: The fitted scaler object (if fit_scaler=True) or the provided scaler.
    """
    print("Preprocessing CAN data (scaling and sequencing)...")
    if scaler is None:
        scaler = MinMaxScaler()
    
    if fit_scaler:
        scaled_data = scaler.fit_transform(can_data)
    else:
        scaled_data = scaler.transform(can_data)

    # Create sequences
    X = create_sequences(scaled_data, sequence_length)
    return X, scaler

def build_autoencoder_model(input_shape):
    """
    Builds an LSTM-based Autoencoder model.

    The encoder compresses the input sequence into a latent space representation.
    The decoder reconstructs the input sequence from this latent space.
    The goal is to learn a compressed representation of "normal" data.

    Args:
        input_shape (tuple): Shape of a single input sequence (sequence_length, num_features).

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
    Trains the Autoencoder model on normal CAN bus data.

    The model learns to reconstruct normal data. Anomalies, when presented
    to the trained model, will result in higher reconstruction errors.

    Args:
        model (tensorflow.keras.Model): The Autoencoder model to train.
        X_train_normal (np.array): Training data containing only normal sequences.
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

def detect_anomalies(model, X_data, threshold_percentile=95):
    """
    Detects anomalies in the input data using the trained Autoencoder.

    Anomalies are identified by high reconstruction errors (MSE) compared to a
    threshold derived from the reconstruction errors of the training data (normal data).

    Args:
        model (tensorflow.keras.Model): The trained Autoencoder model.
        X_data (np.array): Data sequences to check for anomalies.
        threshold_percentile (int): Percentile of reconstruction errors from the *normal training data*
                                     to use as the anomaly threshold.

    Returns:
        tuple: A tuple containing:
            - np.array: Boolean array indicating detected anomalies (True for anomaly).
            - np.array: Reconstruction errors for each sequence in X_data.
            - float: The calculated anomaly threshold.
    """
    print("Detecting anomalies...")
    # Get reconstruction errors for the global training data (normal only) to set a threshold
    global X_train_normal_global
    if X_train_normal_global is None:
        raise ValueError("X_train_normal_global must be set before calling detect_anomalies for threshold calculation.")

    X_train_pred = model.predict(X_train_normal_global)
    train_reconstruction_errors = np.mean(np.square(X_train_normal_global - X_train_pred), axis=(1, 2))

    # Determine threshold based on a percentile of normal reconstruction errors
    # Anomalies should have errors significantly higher than normal data
    threshold = np.percentile(train_reconstruction_errors, threshold_percentile)
    print(f"Anomaly detection threshold set at {threshold_percentile}th percentile of training errors: {threshold:.4f}")

    # Predict and calculate reconstruction errors for the input data (X_data)
    X_data_pred = model.predict(X_data)
    data_reconstruction_errors = np.mean(np.square(X_data - X_data_pred), axis=(1, 2))

    # Flag anomalies
    anomalies = data_reconstruction_errors > threshold
    return anomalies, data_reconstruction_errors, threshold

def main():
    """
    Main function to run the CAN bus anomaly detection prototype.
    Orchestrates data simulation, preprocessing, model building, training, and anomaly detection.
    """
    global X_train_normal_global # Access the global variable

    # --- 1. Simulate CAN Bus Data ---
    sequence_length = 10 # Number of consecutive messages in a window
    raw_can_data, labels = simulate_can_data(num_normal_messages=20000, num_anomalies=200, sequence_length=sequence_length)
    
    # Separate normal data for training the autoencoder.
    # In a real scenario, this would be a large, verified dataset of normal operation.
    # For this prototype, we strictly filter out any simulated anomalies from the training set.
    can_data_normal_only_for_training = raw_can_data[labels == 0]
    
    # Ensure there's enough data for creating sequences after filtering
    if len(can_data_normal_only_for_training) < sequence_length:
        print(f"Not enough normal data ({len(can_data_normal_only_for_training)} messages) to create sequences of length {sequence_length}. Adjust simulation parameters.")
        return

    # --- 2. Preprocess Training Data ---
    # Fit the scaler ONLY on normal training data and store it
    X_train_normal_global, scaler = preprocess_can_data(can_data_normal_only_for_training, sequence_length, fit_scaler=True)
    
    # --- 3. Preprocess All Data for Evaluation ---
    # Apply the *already fitted* scaler to the entire dataset (normal + anomalies)
    # Then create sequences and align labels.
    all_scaled_data = scaler.transform(raw_can_data)
    X_all_sequences = create_sequences(all_scaled_data, sequence_length)

    # Adjust labels to match sequences. If ANY message in a sequence is anomalous,
    # we consider the whole sequence anomalous. This is a simplification for evaluation.
    sequence_labels = np.array([1 if np.any(labels[i:i+sequence_length] == 1) else 0 
                                for i in range(len(raw_can_data) - sequence_length + 1)])

    # Handle cases where create_sequences might return empty if data is too short
    if X_all_sequences.size == 0:
        print("Not enough total data to create sequences for evaluation. Exiting.")
        return
    if X_train_normal_global.size == 0:
        print("Not enough normal training data to create sequences for training. Exiting.")
        return


    # --- 4. Build Autoencoder Model ---
    input_shape = (X_train_normal_global.shape[1], X_train_normal_global.shape[2]) # (sequence_length, num_features)
    model = build_autoencoder_model(input_shape)

    # --- 5. Train Autoencoder ---
    trained_model, history = train_autoencoder(model, X_train_normal_global, epochs=100, batch_size=64)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 6. Detect Anomalies ---
    # Using the globally defined X_train_normal_global for threshold calculation
    anomalies_detected, all_reconstruction_errors, threshold = detect_anomalies(trained_model, X_all_sequences, threshold_percentile=98)

    # --- 7. Evaluation and Visualization ---
    print("\n--- Anomaly Detection Results ---")
    print(f"Total sequences processed: {len(X_all_sequences)}")
    print(f"Total actual anomalies (sequences with at least one anomalous message): {np.sum(sequence_labels == 1)}")
    print(f"Total detected anomalies: {np.sum(anomalies_detected)}")

    # Simple accuracy metrics (be cautious with highly imbalanced datasets)
    true_positives = np.sum((anomalies_detected == True) & (sequence_labels == 1))
    false_positives = np.sum((anomalies_detected == True) & (sequence_labels == 0))
    true_negatives = np.sum((anomalies_detected == False) & (sequence_labels == 0))
    false_negatives = np.sum((anomalies_detected == False) & (sequence_labels == 1))

    print(f"True Positives (correctly identified anomalies): {true_positives}")
    print(f"False Positives (normal data identified as anomaly): {false_positives}")
    print(f"True Negatives (correctly identified normal data): {true_negatives}")
    print(f"False Negatives (anomalies missed): {false_negatives}")

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
    plt.plot(all_reconstruction_errors, marker='o', linestyle='', markersize=2, label='Reconstruction Error')
    
    # Highlight actual anomalies in the plot
    actual_anomaly_indices = np.where(sequence_labels == 1)[0]
    if len(actual_anomaly_indices) > 0:
        plt.plot(actual_anomaly_indices, all_reconstruction_errors[actual_anomaly_indices], 
                 'ro', markersize=4, label='Actual Anomalies')
        
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({threshold_percentile}th percentile)')
    plt.title('CAN Bus Anomaly Detection: Reconstruction Errors')
    plt.xlabel('Sequence Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Prototype Complete ---")
    print("This prototype demonstrates how an LSTM Autoencoder can be used for unsupervised anomaly detection on simulated CAN bus data.")
    print("In a real-world scenario, careful selection of features, extensive normal data collection, and tuning of the anomaly threshold are crucial.")
    print("The anomaly labels generated here are for evaluation purposes of the prototype. Anomaly detection itself is an unsupervised task.")


if __name__ == "__main__":
    main()
