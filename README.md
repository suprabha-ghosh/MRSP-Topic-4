# MRSP-Topic-4

Implementation of a Neural Network-Based Perceptual Loss for Audio
================================================================================

📌 PROJECT OVERVIEW
-------------------
This project implements and evaluates a perceptual loss function for audio 
based on deep neural network embeddings from **OpenL3** and **VGGish**. 
We investigate whether these embeddings can reflect perceptual differences 
between clean and degraded audio, and compare them with traditional 
audio quality metrics — PESQ and STOI.

📌 OBJECTIVES
-------------
- Extract high-level neural audio embeddings using pretrained models
- Measure similarity between clean and degraded audio via cosine similarity
- Compute PESQ and STOI scores for the same audio pairs
- Compare and visualize correlations between neural and traditional metrics

📌 OUTCOME
----------
- Python implementation of full pipeline
- Modular scripts for preprocessing, embedding extraction, and analysis
- Result visualizations and statistical evaluations
- Final analysis report and documentation

================================================================================

📁 PROJECT STRUCTURE
---------------------
audio_conversion.py              → Convert original audio to 16kHz mono
extract_embeddings.py        → Add noise, extract OpenL3 and VGGish embeddings
analyze_results.py            → Compute PESQ/STOI, cosine similarity, visualize

📂 test_audio/                → Original and degraded audio files
📂 openl3_embeddings/         → .npy files for OpenL3 embeddings
📂 vggish_embeddings/         → .npy files for VGGish embeddings
📂 results/                   → Scatter plots, correlation heatmaps, CSVs, report

================================================================================

🛠 DEPENDENCIES
---------------
- Python 3.8+
- librosa
- soundfile
- numpy
- scipy
- openl3
- torch, torchaudio
- torchvggish (https://github.com/harritaylor/torchvggish)
- pesq
- pystoi
- pandas
- matplotlib, seaborn

Install all dependencies with:

    pip install -r requirements.txt

================================================================================

🚀 HOW TO RUN
-------------
1. Convert input audio to 16 kHz mono:

       python audio_conversion.py

2. Generate clean and degraded audio embeddings:

       python extract_embeddings.py

3. Analyze and visualize results:

       python analyze_results.py

Output files will be saved in the `results/` folder.

================================================================================

📊 SAMPLE RESULTS
-----------------
| Noise Level | OpenL3 Similarity | VGGish Similarity | PESQ Score | STOI Score |
|-------------|------------------|-------------------|------------|------------|
| 0.001       | 0.994            | 0.941             | 3.187      | 1.000      |
| 0.020       | 0.978            | 0.864             | 1.066      | 0.963      |

- OpenL3 correlates strongly with STOI (0.947)
- VGGish correlates strongly with PESQ (0.970)

📂 Visualizations:
- Scatter plots of similarity vs PESQ/STOI
- Heatmaps of correlation matrices
- Noise level vs metric line plots
- Bar charts comparing models

📄 Analysis report:
- See `results/analysis_report.txt`

================================================================================

📚 REFERENCES
------------
- Cramer et al., ICASSP 2019 - Look, Listen, and Learn More (OpenL3)
- Hershey et al., ICASSP 2017 - CNNs for AudioSet (VGGish)
- ITU-T P.862 – PESQ standard
- Taal et al., STOI definition
- OpenL3: https://github.com/marl/openl3
- TorchVGGish: https://github.com/harritaylor/torchvggish
- PESQ: https://github.com/ludlows/python-pesq
- PySTOI: https://github.com/mpariente/pystoi

================================================================================

👤 AUTHOR
---------
Name: Suprabha Ghosh  
Matriculation Number: 64365  
Email: suprabha.ghosh@tu-ilmenau.de  

This project was completed as part of Multirate Signal Processing at Technische Unversität Ilmenau.

================================================================================

📂 LICENSE & USAGE
------------------
This project is for academic and educational purposes only.  
All libraries used are under open-source licenses.
