import numpy as np
import struct
from data_generator import DataGenerator

# Generate synthetic data for testing
num_samples = 1000
sequence_length = 10

# Generate random complex numbers
real_parts = np.random.randn(num_samples)
imag_parts = np.random.randn(num_samples)
samples = real_parts + 1j * imag_parts

# Save synthetic data to a .dat file
with open('data/synthetic_data.dat', 'wb') as f:
    for sample in samples:
        f.write(struct.pack('ff', sample.real, sample.imag))

# Test the data generator and model training
batch_size = 20
max_train_samples = 500

train_gen_instance = DataGenerator('data/synthetic_data.dat', batch_size=batch_size, sequence_length=sequence_length, max_samples=max_train_samples, for_training=True)

for epoch in range(1):
    train_gen_instance.reset()
    for step in range(max_train_samples // (batch_size * sequence_length)):
        X_chunk, Y_chunk = next(train_gen_instance)
        print(f"X_chunk shape: {X_chunk.shape}, Y_chunk shape: {Y_chunk.shape}")
        break  # For testing, break after first batch

train_gen_instance.close()
