import os
import pandas as pd
import json

if __name__ == "__main__":
    # Load hyperparameters
    hyperparameters_path = "/opt/ml/input/config/hyperparameters.json"
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)

    # Load training data
    data_path = "/opt/ml/input/data/training/train.csv"
    data = pd.read_csv(data_path)
    print(f"Training data shape: {data.shape}")

    # Dummy training
    print("Training with hyperparameters:", hyperparameters)

    # Save model
    model_output_path = "/opt/ml/model/model.txt"
    with open(model_output_path, "w") as f:
        f.write("Trained model artifact")
    
    print("Training complete.")
