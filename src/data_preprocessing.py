import pandas as pd
from datasets import load_dataset

def load_and_preprocess_data():
    dataset = load_dataset("Kaludi/Customer-Support-Responses")
    return dataset

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    print(dataset)
