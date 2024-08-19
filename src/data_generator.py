
#### `src/data_generator.py`

```python
import os
import numpy as np
import pandas as pd
import struct

def load_data(filepath):
    _, file_extension = os.path.splitext(filepath)
    
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
        samples = df['your_column_name'].values
    elif file_extension == '.dat':
        with open(filepath, 'rb') as f:
            samples = f.readlines()
        samples = np.array([sample.strip() for sample in samples])
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return samples

def count_lines(filepath):
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)

class DataGenerator:
    def __init__(self, filepath, batch_size, sequence_length, max_samples=None, for_training=True):
        self.filepath = filepath
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.for_training = for_training
        self.samples = []
        self.binary_file = open(self.filepath, 'rb')
        self.reset()

    def reset(self):
        self.total_samples_processed = 0
        _, self.file_extension = os.path.splitext(self.filepath)
        print(f"File extension detected: {self.file_extension}")

    def __iter__(self):
        self.binary_file.seek(0)
        self.samples = []
        return self
    
    def close(self):
        if not self.binary_file.closed:
            self.binary_file.close()

    def process_data(self, samples):
        real_parts = []
        imag_parts = []
        for sample in samples:
            try:
                cnum = complex(sample.replace('j', 'j'))
                real_parts.append(np.real(cnum))
                imag_parts.append(np.imag(cnum))
            except ValueError:
                continue

        real_parts = (real_parts - np.mean(real_parts)) / np.std(real_parts)
        imag_parts = (imag_parts - np.mean(imag_parts)) / np.std(imag_parts)

        X = [list(zip(real_parts[i:i+self.sequence_length], imag_parts[i:i+self.sequence_length])) for i in range(len(real_parts) - self.sequence_length)]
        return np.array(X)

    def __next__(self):
        chunksize = self.batch_size * self.sequence_length
        samples = []
        while True:
            binary_data = self.binary_file.read(8)
            if not binary_data:
                break 
            decoded_data = struct.unpack('ff', binary_data)
            if decoded_data[0] == 0 and decoded_data[1] == 0:
                decoded_line = f"0j\n"
            else:
                if decoded_data[1] >= 0:
                    decoded_line = f"{decoded_data[0]}+{decoded_data[1]}j\n"
                else:
                    decoded_line = f"{decoded_data[0]}{decoded_data[1]}j\n"
            samples.append(decoded_line)

            if self.max_samples and self.total_samples_processed >= self.max_samples:
                raise StopIteration
            self.total_samples_processed += 1

            if len(samples) == chunksize:
                X_chunk = self.process_data(samples)
                if self.for_training:
                    return X_chunk, X_chunk
                else:
                    return X_chunk
