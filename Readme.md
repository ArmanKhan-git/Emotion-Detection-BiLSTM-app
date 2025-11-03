# 🎭 Text Emotion Classifier

A deep learning-based text emotion detection system using BiLSTM with Attention mechanism. Predicts emotions from text input with confidence scores.

## 🚀 [Live Demo](your-streamlit-app-link)

---

## Features

- **Deep Learning Model**: BiLSTM with custom Attention layer for accurate predictions
- **Multi-Emotion Detection**: Classifies text into multiple emotion categories
- **Confidence Scores**: Shows probability distribution for all emotions
- **Interactive UI**: Real-time predictions with visual probability chart
- **Easy to Use**: Simply enter text and get instant results

---

## Tech Stack

- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web interface
- **BiLSTM + Attention** - Neural network architecture
- **scikit-learn** - Label encoding and preprocessing

---

## Model Architecture

- **Embedding Layer**: Converts text to dense vectors
- **Bidirectional LSTM**: Captures context from both directions
- **Attention Mechanism**: Focuses on important words
- **Dense Layer**: Final emotion classification

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ArmanKhan-git/Emotion-Detection-BiLSTM-app
cd Emotion-Detection-BiLSTM-app
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---


## Usage Example

1. Enter any text: *"I'm so happy today!"*
2. Click **Predict**
3. View the predicted emotion with confidence scores
4. See probability distribution for all emotions

---

## Emotions Detected

The model can classify text into emotions such as:
- Joy
- Sadness
- Anger
- Fear
- Surprise
- Love
- *and more...*

---


## Future Improvements

- Multi-language support
- Voice input option
- Emotion history tracking
- Export results feature
- API endpoint for integration

---

## Author

**Your Name**
- GitHub: [@ArmanKhan-git](https://github.com/ArmanKhan-git)

---
