# Importing the important modules

import random
import numpy as np
import os

def generate_dataset_fixed_length(length, num_samples):
    dataset = []
    for _ in range(num_samples):
        sequence = [random.randint(0, 1) for _ in range(length)]
        xor_result = sequence[0]
        for i in range(1, length):
            xor_result ^= sequence[i]
        dataset.append((sequence, xor_result))
    return dataset

def generate_dataset_variable_length(max_length, num_samples):
    dataset = []
    for _ in range(num_samples):
        length = random.randint(1, max_length)
        sequence = [random.randint(0, 1) for _ in range(length)]
        xor_result = sequence[0]
        for i in range(1, length):
            xor_result ^= sequence[i]
        dataset.append((sequence, xor_result))
    return dataset

if __name__ == "__main__":
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)

    fixed_length_dataset = generate_dataset_fixed_length(length=50, num_samples=100000)
    np.save(os.path.join(output_dir, "fixed_length_dataset.npy"), fixed_length_dataset)

    variable_length_dataset = generate_dataset_variable_length(max_length=50, num_samples=100000)
    np.save(os.path.join(output_dir, "variable_length_dataset.npy"), variable_length_dataset)
