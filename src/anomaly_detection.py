import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_generator import DataGenerator

def plot_with_intrusions(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length, save_path):
    for idx in range(0, len(all_X_chunk_test), sequence_length):
        sequence_idx = idx // sequence_length
        if all_intrusion_flags[sequence_idx]:
            plt.figure(figsize=(14, 6))
            time_steps = np.arange(idx * sequence_length, (idx + 1) * sequence_length)

            real_part_test = all_X_chunk_test[idx, :, 0].reshape(-1)
            imag_part_test = all_X_chunk_test[idx, :, 1].reshape(-1)
            real_part_pred = all_X_chunk_pred[idx, :, 0].reshape(-1)
            imag_part_pred = all_X_chunk_pred[idx, :, 1].reshape(-1)

            plt.plot(time_steps, real_part_test, 'b-', label='Original Real', linewidth=2)
            plt.plot(time_steps, real_part_pred, 'r--', label='Reconstructed Real', linewidth=2)
            plt.plot(time_steps, imag_part_test, 'g-', label='Original Imag', linewidth=2)
            plt.plot(time_steps, imag_part_pred, 'y--', label='Reconstructed Imag', linewidth=2)

            where_fill = np.full_like(time_steps, True, dtype=bool)
            plt.fill_between(time_steps, -3, 3, where=where_fill, color=(1, 0.5, 0.5), alpha=0.3, label='Intrusion Detected')

            plt.title(f'Original vs Reconstructed with Intrusion (Sequence {sequence_idx})', fontsize=20, fontweight='bold')
            plt.xlabel('Sample Index', fontsize=20, fontweight='bold')
            plt.ylabel('IQ Sample', fontsize=20, fontweight='bold')
            plt.legend(loc='lower right', fontsize=15)

            for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                label.set_fontsize(15)
                label.set_fontweight('bold')

            plt.tight_layout()

            filename = os.path.join(save_path, f'intrusion_sequence_{sequence_idx}.png')
            plt.savefig(filename)
            plt.close()

sequence_length = 10
batch_size = 20

# Load the trained model
model = load_model('rnn_autoencoder_model.h5')

# Initialize the data generator for testing
combined_gen_instance = DataGenerator('data/5G_DL_IQ_with_periodic_jamming_0928_02.dat', batch_size=batch_size, sequence_length=sequence_length, for_training=False)

num_predictions = 100
reconstruction_errors = []
all_X_chunk_test = []
all_X_chunk_pred = []
all_intrusion_flags = []

try:
    for _ in range(num_predictions):
        print('Prediction number:', _)
        X_chunk_test = next(combined_gen_instance)
        X_chunk_pred = model.predict(X_chunk_test)
        chunk_errors = np.mean(np.square(X_chunk_test - X_chunk_pred), axis=1)
        reconstruction_errors.extend(chunk_errors)
        all_X_chunk_test.append(X_chunk_test)
        all_X_chunk_pred.append(X_chunk_pred)
except StopIteration:
    print("All samples processed.")

reconstruction_error = np.array(reconstruction_errors)
max_error_per_sequence = reconstruction_error.reshape(-1, 2).max(axis=1)
error_per_sequence = max_error_per_sequence.reshape(-1, sequence_length).mean(axis=1)
threshold1 = np.percentile(error_per_sequence, 99)
threshold2 = np.percentile(reconstruction_error, 99)
is_intrusion_detected = error_per_sequence > threshold1

flat_error_per_sequence = error_per_sequence.flatten()
for error in flat_error_per_sequence:
    all_intrusion_flags.append(error > threshold1)

all_X_chunk_test = np.concatenate(all_X_chunk_test, axis=0)
all_X_chunk_pred = np.concatenate(all_X_chunk_pred, axis=0)

save_path = 'data/intrusion_detected_plots'
os.makedirs(save_path, exist_ok=True)
plot_with_intrusions(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length, save_path)

jamming_detected = reconstruction_error > threshold2
flattened_jamming_detected = jamming_detected.flatten()

real_part_detected = jamming_detected[:, 0]
imag_part_detected = jamming_detected[:, 1]

real_true_count = np.sum(real_part_detected)
real_false_count = len(real_part_detected) - real_true_count

imag_true_count = np.sum(imag_part_detected)
imag_false_count = len(imag_part_detected) - imag_true_count

overall_true_count = np.sum(flattened_jamming_detected)
overall_false_count = len(flattened_jamming_detected) - overall_true_count

df = pd.DataFrame({
    'Part': ['Real', 'Imaginary', 'Overall'],
    'True Count': [real_true_count, imag_true_count, overall_true_count],
    'False Count': [real_false_count, imag_false_count, overall_false_count]
})

print(df)

num_jamming_detected = np.sum(jamming_detected)
print(f"Number of jamming sequences detected: {num_jamming_detected} out of {len(flattened_jamming_detected)} sequences")

plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

reconstruction_error_real = reconstruction_error[:, 0]
reconstruction_error_imag = reconstruction_error[:, 1]

plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error_real, label='Reconstruction Error - Real Part', color='blue')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Real Part with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()
