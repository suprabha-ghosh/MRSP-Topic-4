# 🎵 MRSP-Topic-4: Neural Network-Based Perceptual Loss for Audio

---

## 📝 Project Overview

This project implements and evaluates a **perceptual loss function for audio** using deep neural network embeddings from **OpenL3** and **VGGish**. The goal is to determine whether these embeddings can reflect perceptual differences between clean and degraded audio, and compare them with traditional audio quality metrics — **PESQ** and **STOI**.

---

## 🎯 Objectives

- **Extract** high-level neural audio embeddings using pretrained models (OpenL3, VGGish)
- **Simulate** audio degradations (noise, bitrate, compression)
- **Measure** similarity between clean and degraded audio via cosine similarity
- **Compute** PESQ and STOI scores for the same audio pairs
- **Compare & Visualize** correlations between neural and traditional metrics

---

## 🏆 Outcomes

- Modular Python scripts for preprocessing, embedding extraction, and analysis
- Result visualizations and statistical evaluations
- Final analysis report and documentation

---

## 📂 Project Structure

```
audio_conversion.py      # Convert original audio to 16kHz mono
extract_embeddings.py    # Add degradations, extract OpenL3 and VGGish embeddings
calculate_metrics.py     # Compute PESQ/STOI, cosine similarity, visualize
visualize_results.py     # Generate plots, tables, analysis report

original_audio/          # Original raw audio files
test_audio/              # 16Hz mono and degraded audio files
openl3_embeddings/       # .npy files for OpenL3 embeddings
vggish_embeddings/       # .npy files for VGGish embeddings
results/                 # Scatter plots, correlation heatmaps, CSVs, report
requirements.txt         # Python dependencies
README.md                # Project documentation
```

---

## ⚙️ Dependencies

- Python 3.8+
- [librosa](https://librosa.org/)
- [soundfile](https://pysoundfile.readthedocs.io/)
- numpy
- scipy
- [openl3](https://github.com/marl/openl3)
- torch, torchaudio
- [torchvggish](https://github.com/harritaylor/torchvggish)
- [pesq](https://github.com/ludlows/python-pesq)
- [pystoi](https://github.com/mpariente/pystoi)
- pandas
- matplotlib, seaborn

**Install all dependencies:**
```sh
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Usage Guide

### 1️⃣ Convert Input Audio

Convert original audio files to 16 kHz mono format:
```sh
python audio_conversion.py --input original_audio --output test_audio --sr 16000
```

### 2️⃣ Generate Embeddings & Degraded Audio

Create clean and degraded audio, extract OpenL3 and VGGish embeddings:
```sh
python extract_embeddings.py --input-dir test_audio --openl3-dir openl3_embeddings --vggish-dir vggish_embeddings
```

### 3️⃣ Calculate Metrics

Compute PESQ/STOI scores and cosine similarities:
```sh
python calculate_metrics.py --audio-dir test_audio --openl3-dir openl3_embeddings --vggish-dir vggish_embeddings --results-dir results
```

### 4️⃣ Visualize & Analyze Results

Generate plots, tables, and analysis report:
```sh
python visualize_results.py --results-dir results --output-dir results/visualizations
```

**Output files** will be saved in the `results/` and `results/visualizations/` folders.

---

## 📊 Sample Results

| Degradation Type      | OpenL3 Similarity | VGGish Similarity | PESQ Score | STOI Score |
|---------------------- |------------------|-------------------|------------|------------|
| Noise (Low)           | 0.992            | 0.907             | 2.466      | 0.999      |
| Noise (High)          | 0.985            | 0.882             | 1.199      | 0.988      |
| Noise (Extreme)       | 0.978            | 0.859             | 1.066      | 0.963      |
| Bitrate (Low)         | 0.991            | 0.958             | 3.656      | 0.995      |
| Compression (High)    | 0.989            | 0.964             | 4.312      | 1.000      |
| ...                   | ...              | ...               | ...        | ...        |

- **OpenL3** correlates strongly with **STOI** (0.947)
- **VGGish** correlates strongly with **PESQ** (0.970)

---

## 📈 Visualizations & Reports

- **Scatter plots**: Similarity vs PESQ/STOI
- **Heatmaps**: Correlation matrices
- **Bar charts**: Metric comparisons by degradation type
- **Analysis report**: See [`results/visualizations/analysis_report.txt`](results/visualizations/analysis_report.txt)

---

## 📚 References

- Cramer et al., ICASSP 2019 - Look, Listen, and Learn More ([OpenL3](https://github.com/marl/openl3))
- Hershey et al., ICASSP 2017 - CNNs for AudioSet ([VGGish](https://github.com/harritaylor/torchvggish))
- ITU-T P.862 – PESQ standard ([PESQ](https://github.com/ludlows/python-pesq))
- Taal et al., STOI definition ([PySTOI](https://github.com/mpariente/pystoi))
- [librosa](https://librosa.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

---

## 👤 Author

**Suprabha Ghosh**  
Matriculation Number: 64365  
Email: suprabha.ghosh@tu-ilmenau.de  
Technische Universität Ilmenau  
*Project completed as part of Multirate Signal Processing.*

---

## 📄 License & Usage

This project is for **academic and educational purposes only**.  
All libraries used are under open-source licenses.

---
