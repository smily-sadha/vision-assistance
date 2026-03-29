import os
import numpy as np
from model import distance_model

def calculate_distance(embedding1, embedding2):
    return distance_model.predict(np.array([embedding1, embedding2]))

if __name__ == "__main__":
    # Example usage
    embedding1 = [0.1, 0.2, 0.3]
    embedding2 = [0.4, 0.5, 0.6]
    distance = calculate_distance(embedding1, embedding2)
    print(f"Distance: {distance}")