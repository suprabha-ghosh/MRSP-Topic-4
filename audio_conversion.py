import os
import librosa
import soundfile as sf

# Input and output directories
input_dir = "original_audio"
output_dir = "test_audio"
os.makedirs(output_dir, exist_ok=True)

# Target sample rate
TARGET_SR = 16000

# Loop through files
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        sf.write(output_path, y, TARGET_SR)
        print("Converted:", output_path)

print("Batch conversion complete.")
