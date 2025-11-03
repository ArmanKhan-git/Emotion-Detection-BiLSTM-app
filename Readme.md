🎭 Text Emotion ClassifierThis is a Streamlit web application that uses a deep learning model to predict the emotion from a piece of text.🚀 Live DemoClick here to try the app!The app uses a Bidirectional LSTM (Bi-LSTM) model with a custom Attention mechanism. This allows the model to not only understand the context of a sentence but also to "pay attention" to the specific words that are most important for determining the emotional tone (e.g., "betray" in "I can't believe you would betray me").✨ FeaturesReal-time Emotion Prediction: Enter any text and get an instant classification.Advanced NLP Model: Built with a Bi-LSTM and a custom Attention layer for high-accuracy contextual understanding.Full Probability Breakdown: A clean bar chart shows the model's confidence score for all possible emotions.Interactive UI: A simple and clean user interface built with Streamlit.🛠 Tech StackFrontend: StreamlitDeep Learning: TensorFlow & KerasModel Architecture: Bidirectional LSTM + Custom Attention LayerData Processing: Pandas, NumPyPreprocessing: Scikit-learn (for LabelEncoder), Tokenizer🚀 How to Run Locally1. PrerequisitesYou must have the following pre-trained files in the same directory as app.py. These files are generated during the model training process.emotion_lstm_model.h5: The saved and trained Keras model.tokenizer.pkl: The saved Keras Tokenizer object.label_encoder.pkl: The saved Scikit-learn LabelEncoder object.2. Clone the Repositorygit clone <your-repo-url>
cd <your-repo-name>

3. Create a Virtual Environment# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install DependenciesInstall all the required libraries from the requirements.txt file.pip install -r requirements.txt

5. Run the Appstreamlit run app.py

