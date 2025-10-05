# Tycho-AI-Trainer
AI model for detecting exoplanets

## Summary
Our project uses deep learning for exoplanet discovery through data analysis of the Kepler Objects of Interest (KOI) dataset. We developed a Convolutional Neural Network (CNN) capable of identifying subtle patterns in the stellar features that indicate the possible presence of an exoplanet. The program processes large volumes of astronomical data automatically and much faster than manual methods. It also allows hyperparameters to be adjusted through a web interface to improve accuracy and generate new models. This tool contributes to the search for exoplanets by facilitating the detection of promising candidates and helping astronomers prioritize future observations. Its importance lies in demonstrating how artificial intelligence (AI) can accelerate the discovery of new worlds.

## How to execute
#### Linux
```bash
python -m venv .venv
source .venv/bin/activate

python App.py
```
#### Windows
```bash
python -m venv .venv
./.venv/bin/activate
python App.py
```

## Python libraries 
Flask, TensorFlow, Numpy, Pandas, Scikit-learn, Matplotlib
