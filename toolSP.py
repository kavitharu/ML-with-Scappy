import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scapy.all import sniff, IP
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Preprocess the Dataset
def preprocess_data(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Fill missing values in the feature columns with 0
    df.fillna(0, inplace=True)  # Ensuring no NaN values in features
    
    # Map 'classification' column to binary values (malicious = 1, benign = 0)
    df['classification'] = df['classification'].map({'malicious': 1, 'benign': 0})
    
    # Drop rows where 'classification' is NaN (if any remain)
    df.dropna(subset=['classification'], inplace=True)
    
    # Drop the 'hash' column (assuming it's not relevant)
    if 'hash' in df.columns:
        df.drop(columns=['hash'], inplace=True)
    
    # Separate features and labels
    X = df.drop(columns=['classification'])
    y = df['classification']
    
    return X, y

# Step 3: Train-Test Split and Feature Scaling
def split_and_scale_data(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Step 4: Build and Train the Machine Learning Model
def build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    # Build a simple DNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return model, history

# Step 5: Plot Training History
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Step 6: Evaluate the Model
def evaluate_model(model, X_test_scaled, y_test):
    # Check for any NaN values in y_test
    if y_test.isnull().any():
        print("Warning: y_test contains NaN values. Dropping NaNs.")
        y_test.dropna(inplace=True)

    # Get predictions for the test set
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

# Step 7: Real-Time Packet Capture and Malware Detection (Optional)
def detect_malware(packet, model, scaler, X_train):
    if packet.haslayer(IP):
        length = len(packet)
        protocol = packet[IP].proto
        
        # Create a DataFrame for this single packet with dummy values
        packet_data = pd.DataFrame([{
            "millisecond": packet.time,  # Packet capture time
            "state": 0,  # Example placeholder data
            "usage_counter": 0,  # Example placeholder data
            "prio": 0,
            "static_prio": 0,
            "normal_prio": 0,
            "policy": 0,
            "vm_pgoff": 0
        }])
        
        # Standardize features (reindex to match training data format)
        packet_data = packet_data.reindex(columns=X_train.columns, fill_value=0)
        packet_data_scaled = scaler.transform(packet_data)
        
        # Predict using the trained model
        prediction = model.predict(packet_data_scaled)
        if prediction >= 0.5:
            print("Malicious packet detected!")
        else:
            print("Packet is benign.")

# Main Program
def main():
    # Specify the path to the dataset
    dataset_path = "C:/Users/kthar/Downloads/archive (4)/Malware dataset.csv"
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(dataset_path)
    
    # Step 3: Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Step 4: Build and train model
    model, history = build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 5: Plot training history
    plot_training_history(history)
    
    # Step 6: Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Step 7: Real-time malware detection (Optional)
    print("Sniffing packets for real-time malware detection (Ctrl+C to stop)...")
    sniff(prn=lambda packet: detect_malware(packet, model, scaler, X), count=10)  # Modify `count` as needed

# Run the main program
if __name__ == "__main__":
    main()
