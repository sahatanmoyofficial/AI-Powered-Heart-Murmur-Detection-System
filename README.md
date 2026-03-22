# 🫀 Heart Murmur Detection Using Audio Signals & Deep Learning (HMAD)

> **An end-to-end deep learning system that classifies heart sounds into 5 categories — Artifact, Murmur, Normal, Extrahls, and Extrastole — using MFCC feature extraction from `.wav` recordings and an LSTM model deployed via Streamlit with the model hosted on Hugging Face Hub**
>
> Audio signal processing meets medical AI: `.wav` heart recordings → librosa resampling at 22,050 Hz → 52-coefficient MFCC extraction → LSTM sequence model → 5-class prediction → Streamlit web interface with waveform visualisation. Class imbalance handled via weighted loss. Model served dynamically from `sahatanmoyofficial/lstm_model.keras` on Hugging Face Hub.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace%20Hub-yellow)](https://huggingface.co/sahatanmoyofficial/lstm_model.keras)
[![Librosa](https://img.shields.io/badge/Audio-Librosa%200.11.0-green)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1xAxZJtcZn8tCX2whnP-KEm1zfMIV1phO/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [System Architecture](#4-system-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset](#6-dataset) |
| 7 | [Audio Preprocessing Pipeline](#7-audio-preprocessing-pipeline) |
| 8 | [MFCC Feature Extraction](#8-mfcc-feature-extraction) |
| 9 | [LSTM Model Architecture](#9-lstm-model-architecture) |
| 10 | [Class Imbalance Handling](#10-class-imbalance-handling) |
| 11 | [Model Hosting — Hugging Face Hub](#11-model-hosting--hugging-face-hub) |
| 12 | [Streamlit Web Application](#12-streamlit-web-application) |
| 13 | [How to Replicate](#13-how-to-replicate) |
| 14 | [Business Applications & Other Domains](#14-business-applications--other-domains) |
| 15 | [How to Improve This Project](#15-how-to-improve-this-project) |
| 16 | [Troubleshooting](#16-troubleshooting) |
| 17 | [Glossary](#17-glossary) |

---

## 1. Problem Statement

### What problem are we solving?

Heart murmurs are abnormal sounds produced during the cardiac cycle — caused by turbulent blood flow — that may indicate serious underlying conditions including valve disease, congenital heart defects, and anaemia. Traditional detection requires trained cardiologists with stethoscopes, making screening in resource-limited settings difficult.

Automated analysis of heart sound recordings using machine learning enables:
- **Scalable screening** — process hundreds of recordings without specialist availability
- **Objective classification** — remove inter-rater variability between clinicians
- **Remote monitoring** — integrate with digital stethoscopes and telehealth platforms
- **Triage support** — flag high-risk recordings for specialist review

### What does HMAD classify?

> *"Given a heart sound recording — is this Normal, a Murmur, an Extrasystole (Extrastole), an Extra Heart Sound (Extrahls), or an Artifact?"*

### The 5 Classes

| Class | Description |
|-------|-------------|
| **Normal** | Regular S1/S2 heart sounds with no abnormality |
| **Murmur** | Abnormal whooshing sound due to turbulent blood flow |
| **Extrastole** | Extra heartbeat (premature contraction) |
| **Extrahls** | Additional heart sound (S3/S4 gallop) |
| **Artifact** | Non-cardiac noise from recording artefacts or movement |

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | Multi-class audio classification (5 classes) |
| **Input** | `.wav` or `.mp3` heart sound recordings |
| **Audio standardisation** | Resample to 22,050 Hz via librosa |
| **Feature extraction** | 52 MFCC coefficients → mean over time → shape `(1, 52, 1)` |
| **Model** | LSTM (Keras / TensorFlow 2.20.0) |
| **Imbalance handling** | Class weights during training |
| **Model storage** | Hugging Face Hub: `sahatanmoyofficial/lstm_model.keras` |
| **Model loading** | `hf_hub_download` + `tf.keras.models.load_model` + `@st.cache_resource` |
| **Frontend** | Streamlit — file upload, waveform plot, prediction output |
| **Deployment** | Streamlit Cloud |
| **Test files** | `artifact__201106110909.wav`, `murmur__193_1308078104592_B.wav` |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10 | Core language |
| **Audio loading** | librosa 0.11.0 | Load, resample, and process `.wav`/`.mp3` files |
| **Audio file I/O** | soundfile 0.13.1 | Low-level audio file reading |
| **Feature extraction** | librosa MFCC | Extract 52 Mel-frequency cepstral coefficients |
| **Numeric computing** | NumPy 2.2.6 | Array operations, `np.mean`, `np.expand_dims` |
| **Data handling** | Pandas 2.3.3 | Dataset management and analysis |
| **Deep Learning** | TensorFlow 2.20.0 / Keras | LSTM model definition, training, and inference |
| **Model hosting** | Hugging Face Hub | Remote model storage and versioned downloads |
| **Model download** | `huggingface_hub.hf_hub_download` | Downloads `.keras` model file at runtime |
| **Web framework** | Streamlit | Interactive file upload + prediction UI |
| **Visualisation** | Matplotlib 3.10.8 + `librosa.display.waveshow` | Waveform plotting in the app |
| **Logging** | Python `logging` | Per-module named loggers with timestamp formatting |

---

## 4. System Architecture

```
User uploads .wav / .mp3
         │
┌────────▼──────────────────────────────────────────────────────────┐
│                    AUDIO PREPROCESSING                             │
│                                                                    │
│  librosa.load(uploaded_file, sr=22050)                            │
│    → Resamples to 22,050 Hz                                       │
│    → Returns: y (float32 NumPy array), sr (int)                   │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│                    MFCC FEATURE EXTRACTION                         │
│                                                                    │
│  librosa.feature.mfcc(y=y, sr=sr, n_mfcc=52)                     │
│    → shape: (52, T) where T = time frames                         │
│  np.mean(mfcc.T, axis=0)                                          │
│    → shape: (52,) — temporal average of each coefficient          │
│  np.expand_dims(..., axis=0) → shape: (1, 52)                     │
│  np.expand_dims(..., axis=2) → shape: (1, 52, 1)  ← LSTM input   │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│                    LSTM MODEL (Keras)                              │
│                                                                    │
│  Loaded from: sahatanmoyofficial/lstm_model.keras (HF Hub)        │
│  @st.cache_resource — loaded once, cached across all sessions     │
│                                                                    │
│  input shape: (batch, 52, 1)                                      │
│  output: softmax over 5 classes                                   │
│  prediction = model.predict(X_input) → shape (1, 5)              │
│  predicted_class = np.argmax(prediction, axis=1)[0]               │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                                    │
│                                                                    │
│  • File uploader (.wav / .mp3)                                    │
│  • Waveform plot (librosa.display.waveshow via Matplotlib)        │
│  • Predicted class index (0–4)                                    │
│  • Raw softmax scores array                                       │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

```
Heart-Murmur-Detection/
│
├── app.py                    # Streamlit application entry point
│
├── audio/
│   ├── __init__.py
│   └── preprocessing.py      # load_audio() · extract_mfcc()
│
├── model/
│   ├── __init__.py
│   └── model_loader.py       # load_model() — HF Hub download + @st.cache_resource
│
├── ui/
│   ├── __init__.py
│   └── visualizations.py     # plot_waveform() — librosa.display.waveshow
│
├── utils/
│   ├── __init__.py
│   └── logger.py             # setup_logger() — named loggers with StreamHandler
│
├── config.py                 # SAMPLE_RATE=22050 · N_MFCC=52 · HF repo ID
├── versions.py               # Utility script to print installed package versions
├── requirements.txt          # All dependencies with pinned versions
│
└── testfiles/
    ├── artifact__201106110909.wav       # Test file: artifact class
    └── murmur__193_1308078104592_B.wav  # Test file: murmur class
```

---

## 6. Dataset

### Heartbeat Sound Dataset (Kaggle)

| Property | Detail |
|----------|--------|
| **Source** | Kaggle — [Heartbeat Sound Dataset](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound) |
| **Format** | `.wav` audio files |
| **License** | Open source |
| **Classes** | 5 labelled classes + unlabelled files (used for inference only) |

### Class Distribution (Imbalanced)

The dataset is naturally imbalanced — normal heart sounds are far more common than pathological ones in a general population recording:

| Class | Label | Notes |
|-------|-------|-------|
| **Artifact** | 0 | Recording noise, movement artefacts |
| **Extrastole** | 1 | Premature cardiac contractions |
| **Extrahls** | 2 | Additional heart sounds (S3/S4 gallop) |
| **Murmur** | 3 | Turbulent blood flow sounds |
| **Normal** | 4 | Regular S1/S2 with no pathology |

> **Note:** The predicted class indices (0–4) output by the current app are not mapped to class names in the UI — see Section 15 for the improvement.

### Test Files Included

Two test recordings are included in `testfiles/`:
- `artifact__201106110909.wav` — demonstrates the Artifact class
- `murmur__193_1308078104592_B.wav` — demonstrates the Murmur class

---

## 7. Audio Preprocessing Pipeline

### `audio/preprocessing.py` — `load_audio()`

```python
def load_audio(uploaded_file):
    y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)  # sr=22050
    return y, sr
```

`librosa.load` with an explicit `sr` parameter resamples the input audio to exactly 22,050 Hz regardless of the original recording rate. This standardisation is critical: MFCC feature dimensions depend on the sampling rate, so inconsistent input rates would produce incompatible features for the LSTM.

**What it returns:**
- `y` — `float32` NumPy array of audio samples (values normalised to `[-1.0, +1.0]`)
- `sr` — always `22050` (int) — the target sampling rate

---

## 8. MFCC Feature Extraction

### What are MFCCs?

Mel-Frequency Cepstral Coefficients (MFCCs) capture the short-term power spectrum of a sound, mapped onto the Mel perceptual scale. They compress the key spectral characteristics of audio into a compact feature vector — widely used in speech recognition, music information retrieval, and medical audio analysis. For heart sounds, MFCCs capture the frequency patterns that distinguish normal lub-dub from murmurs, gallops, and artefacts.

### `audio/preprocessing.py` — `extract_mfcc()`

```python
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # N_MFCC=52
    # mfcc shape: (52, T) — 52 coefficients across T time frames

    mfcc_scaled = np.mean(mfcc.T, axis=0)
    # mfcc.T shape: (T, 52) — transpose
    # np.mean(..., axis=0) → shape: (52,) — average each coeff over time

    X_input = np.expand_dims(mfcc_scaled, axis=0)   # (1, 52)
    X_input = np.expand_dims(X_input, axis=2)        # (1, 52, 1)

    return X_input
```

### Shape Pipeline

```
Raw audio (y)
    ↓
librosa.feature.mfcc(n_mfcc=52)
    ↓ shape: (52, T)  where T varies with audio duration
np.mean(mfcc.T, axis=0)
    ↓ shape: (52,)    temporal average — reduces variable length to fixed 52 values
np.expand_dims(axis=0)
    ↓ shape: (1, 52)  add batch dimension
np.expand_dims(axis=2)
    ↓ shape: (1, 52, 1)  add channel dimension for LSTM input
```

### Why 52 Coefficients?

`N_MFCC=52` is set in `config.py`. Standard MFCC implementations use 13–40 coefficients. 52 provides a richer representation of the heart sound's spectral structure — capturing fine-grained frequency patterns across the cardiac cycle — at the cost of higher input dimensionality.

### Temporal Averaging: A Design Choice

Computing `np.mean(mfcc.T, axis=0)` reduces the time dimension to a single mean vector. This makes the feature representation **independent of audio length** — important because heart sound recordings vary in duration. The trade-off is loss of temporal dynamics within the recording. A more sophisticated approach would preserve the time dimension for the LSTM (see Section 15).

---

## 9. LSTM Model Architecture

### What is an LSTM?

Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) designed to model sequential data. They use gating mechanisms (input gate, forget gate, output gate) to selectively retain or discard information across time steps. For audio classification, LSTMs process the sequence of features across time.

### Input Shape

```
X_input shape: (batch, timesteps, features) = (1, 52, 1)
```

The model treats the 52 MFCC coefficients as 52 time steps, each with 1 feature. This framing presents each MFCC coefficient as one temporal step in a sequence — allowing the LSTM to learn relationships between frequency bands.

### Model Loading (`model/model_loader.py`)

```python
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,            # "sahatanmoyofficial/lstm_model.keras"
        filename=HF_MODEL_FILENAME,    # "lstm_model.keras"
        repo_type="model"
    )
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

**Key design decisions:**
- `@st.cache_resource` — caches the loaded model across all Streamlit user sessions; model is downloaded and loaded exactly once per deployment, not once per user
- `hf_hub_download` — downloads the `.keras` model file from Hugging Face Hub to the local cache, with automatic version management
- `model.compile()` after loading — ensures the model is ready for both prediction and any evaluation calls

---

## 10. Class Imbalance Handling

The dataset has significantly more normal heart sounds than pathological ones. Training without correction biases the model toward predicting "Normal" for all inputs (achieving high accuracy while missing all murmurs).

### Class Weights Approach

Instead of resampling (oversampling/undersampling), **class weights** are applied during training:

```python
# Conceptual — training code not included in this inference repo
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

model.fit(X_train, y_train, class_weight=class_weight_dict)
```

**Why class weights over resampling for medical data?**

| Approach | Advantage | Disadvantage |
|----------|-----------|-------------|
| **Class weights** | Preserves original data · No synthetic samples · Suitable for small medical datasets | Doesn't add new training examples |
| **SMOTE oversampling** | Adds synthetic minority examples | Risk of overfitting on synthetic audio features |
| **Undersampling** | Balances classes | Discards real majority data |

Class weights penalise misclassifying rare pathological classes more than misclassifying the common "Normal" class — directly aligning training with clinical priorities (missing a murmur is more costly than misidentifying a normal recording).

---

## 11. Model Hosting — Hugging Face Hub

The trained LSTM model is stored on Hugging Face Hub rather than bundled in the repository, providing:

| Benefit | Detail |
|---------|--------|
| **Version control** | Each upload creates a versioned model checkpoint |
| **No large files in Git** | `.keras` files are typically tens to hundreds of MB |
| **Dynamic loading** | `hf_hub_download` with local caching — subsequent requests skip re-download |
| **Sharable** | Anyone with the `repo_id` can download the same model |

### Configuration (`config.py`)

```python
HF_REPO_ID = "sahatanmoyofficial/lstm_model.keras"
HF_MODEL_FILENAME = "lstm_model.keras"
```

### Download Flow

```
First run:
  hf_hub_download() → downloads from HuggingFace → saves to ~/.cache/huggingface/
  tf.keras.models.load_model(local_path) → model in memory
  @st.cache_resource → cached for all sessions

Subsequent runs:
  hf_hub_download() → finds local cache → returns cached path (no network request)
  tf.keras.models.load_model(cached_path) → immediate load
```

---

## 12. Streamlit Web Application

### `app.py` Flow

```python
model = load_model()                     # 1. Download + cache from HF Hub

uploaded_file = st.file_uploader(...)   # 2. User uploads .wav/.mp3

y, sr = load_audio(uploaded_file)        # 3. Resample to 22050 Hz

fig = plot_waveform(y, sr)              # 4. Visualise waveform
st.pyplot(fig)

X_input = extract_mfcc(y, sr)           # 5. Extract 52 MFCCs → (1, 52, 1)

prediction = model.predict(X_input)     # 6. LSTM inference → shape (1, 5)
predicted_class = np.argmax(prediction, axis=1)[0]  # 7. Class index 0–4

st.write(f"Predicted Class: {predicted_class}")     # 8. Display result
st.write("Raw Prediction Scores:", prediction)
```

### Running Locally

```bash
streamlit run app.py
# Opens http://localhost:8501
```

### App Features

| Feature | Implementation |
|---------|---------------|
| **File upload** | `st.file_uploader(type=["wav", "mp3"])` |
| **Waveform visualisation** | `librosa.display.waveshow` via Matplotlib, rendered with `st.pyplot()` |
| **Model inference** | `model.predict()` on `(1, 52, 1)` MFCC tensor |
| **Results display** | Predicted class index + raw softmax scores array |
| **Error handling** | try/except on audio loading, MFCC extraction, and inference — `st.error()` on failure |
| **Logging** | Each step logged via named module loggers |

### Current Limitation: Class Names Not Displayed

The app currently shows the integer class index (`0`, `1`, `2`, `3`, or `4`) rather than the class name. Adding a label map is a high-priority improvement:

```python
# Add to app.py
CLASS_NAMES = {0: "Artifact", 1: "Extrastole", 2: "Extrahls", 3: "Murmur", 4: "Normal"}
st.write(f"Predicted Class: **{CLASS_NAMES[predicted_class]}**")
```

---

## 13. How to Replicate

### Prerequisites

- Python 3.10+
- Internet connection (first run downloads model from HuggingFace)

---

### Step 1 — Clone and Install

```bash
git clone https://github.com/sahatanmoyofficial/Heart-Murmur-Detection.git
cd Heart-Murmur-Detection

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

### Step 2 — Run the App

```bash
streamlit run app.py
# Opens http://localhost:8501
```

---

### Step 3 — Test with Included Files

Upload either of the test recordings from `testfiles/`:
- `artifact__201106110909.wav` — should predict class 0 (Artifact)
- `murmur__193_1308078104592_B.wav` — should predict class 3 (Murmur)

---

### Step 4 — Check Installed Versions

```bash
python versions.py
# Prints all installed package versions and Python version
```

---

## 14. Business Applications & Other Domains

### Primary Use Cases

| Stakeholder | Application |
|-------------|------------|
| **Rural healthcare clinics** | Screen patients without specialist cardiologists — flag recordings for remote review |
| **Digital stethoscope manufacturers** | Embed real-time classification in Bluetooth stethoscope firmware |
| **Telehealth platforms** | Automate triage for cardiac symptoms reported by remote patients |
| **Paediatric screening** | School-based heart murmur screening at scale |
| **Insurance actuarial systems** | Risk scoring from cardiac audio in underwriting |

### The Architecture Generalises to Other Audio Classification Tasks

| Domain | Classes | Audio Input |
|--------|---------|-------------|
| **Lung sound analysis** | Normal, Crackle, Wheeze, Rhonchus | Chest auscultation |
| **Baby cry analysis** | Hunger, Pain, Discomfort, Fatigue | Infant audio |
| **Environmental sound** | Urban, Nature, Industrial | Ambient recordings |
| **Machinery fault detection** | Normal, Bearing fault, Gear fault | Industrial sensor audio |
| **Speech emotion** | Happy, Sad, Angry, Neutral, Fearful | Voice recordings |

---

## 15. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Add class name mapping in UI** | 🔴 High | Map integer output (0–4) to readable labels: Artifact, Extrastole, Extrahls, Murmur, Normal |
| **Preserve time dimension for LSTM** | 🔴 High | Instead of `np.mean()`, pass the full `(T, 52)` MFCC matrix as a sequence — fix duration first (e.g., pad/trim to 10 seconds) so T is constant; this allows the LSTM to actually model temporal dynamics |
| **Add CNN-Spectrogram model** | 🟡 Medium | Convert audio to Mel-spectrogram image (2D) and use CNN — CNNs on spectrograms often outperform MFCC+LSTM for audio classification |
| **Add precision, recall, F1 per class** | 🟡 Medium | Accuracy alone is misleading on imbalanced medical data — report per-class metrics especially for Murmur and Extrastole |
| **Add confidence threshold** | 🟡 Medium | Only show a prediction if `max(softmax) > threshold` (e.g., 0.7); display "Uncertain — please consult a cardiologist" otherwise |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Standardise audio duration** | Add `librosa.util.fix_length(y, size=sr*10)` to enforce 10-second clips — prevents variable T in MFCC extraction |
| **Add confidence bar chart** | Visualise softmax scores as a Streamlit bar chart (`st.bar_chart`) per class — much more informative than a raw array |
| **Add audio playback** | `st.audio(uploaded_file)` — let users listen to the recording alongside the prediction |
| **Add REST API** | FastAPI `/predict` endpoint for non-Streamlit clients (mobile apps, EHR integrations) |
| **Add LIME/SHAP explainability** | Show which MFCC coefficients drove the prediction — critical for clinical trust |

---

## 16. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `HfHubHTTPError` on model download | Check internet connection; verify `HF_REPO_ID = "sahatanmoyofficial/lstm_model.keras"` in `config.py` is correct |
| `librosa.load` error on upload | File may be corrupted or in an unsupported format — test with the included `.wav` files in `testfiles/` |
| `TF_ENABLE_ONEDNN_OPTS` warning | Already suppressed in `config.py` via `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` |
| Model predicts class 0 for everything | Possible class imbalance not handled — check class weights were applied during training |
| `streamlit: command not found` | Run `pip install streamlit` or activate the virtual environment |
| `ImportError: soundfile` | Run `pip install soundfile==0.13.1` — needed by librosa for `.wav` loading |
| Very slow first prediction | First run downloads the model from HuggingFace — subsequent runs use the local cache; expected behaviour with `@st.cache_resource` |

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **Heart murmur** | An abnormal sound between heartbeats caused by turbulent blood flow, audible with a stethoscope |
| **S1/S2** | The two normal heart sounds — S1 ("lub") when the mitral and tricuspid valves close, S2 ("dub") when the aortic and pulmonary valves close |
| **Extrastole** | A premature heartbeat (extra contraction) that disrupts the normal rhythm |
| **Extrahls** | Additional heart sounds (S3/S4 gallop) indicating reduced ventricular compliance |
| **MFCC** | Mel-Frequency Cepstral Coefficient — a compact representation of the short-term power spectrum of audio, mapped to the Mel perceptual scale |
| **Mel scale** | A perceptual scale of pitch based on human hearing — equal distances on the Mel scale represent equal perceived pitch differences |
| **LSTM** | Long Short-Term Memory — a Recurrent Neural Network variant with gating mechanisms that learn long-range temporal dependencies |
| **`@st.cache_resource`** | Streamlit decorator that caches a resource (model, database connection) across all sessions and re-runs — prevents model re-downloading on every user interaction |
| **`hf_hub_download`** | Hugging Face Hub function that downloads a model file to local cache with version control |
| **`librosa.load`** | Function that loads an audio file and optionally resamples it to a target sampling rate |
| **`librosa.feature.mfcc`** | Extracts N MFCC features from an audio signal using a Short-Time Fourier Transform + Mel filterbank |
| **Sampling rate** | Number of audio samples per second — 22,050 Hz is half of CD quality (44,100 Hz) |
| **`np.expand_dims`** | NumPy function that inserts a new axis at a specified position — used to add batch and channel dimensions |
| **Softmax** | Activation function that converts raw model scores to a probability distribution summing to 1.0 |
| **Class weights** | Training parameter that multiplies the loss for each class — higher weight for minority classes forces the model to prioritise their correct classification |
| **Spectrogram** | A 2D visual representation of audio showing frequency content over time — alternative to MFCC for CNN-based audio classification |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⚠️ **Medical Disclaimer (repeated):** This project is for **educational and research purposes only**. It must not be used for clinical diagnosis, medical decision-making, or patient care. Always consult a qualified healthcare professional for cardiac assessment.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

