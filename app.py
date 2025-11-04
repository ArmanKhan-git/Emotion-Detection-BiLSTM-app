import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Attention mechanism
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        return super(Attention, self).get_config()


# Streamlit UI
st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")
st.title("🎭 Text Emotion Classifier")
st.write("Enter any text below and let the model predict the **emotion**.")

# Load model, tokenizer, and label encoder with error handling
@st.cache_resource
def load_resources():
    try:
        # Try loading the model with custom objects
        model = load_model(
            "emotion_bilstm_model.h5", 
            custom_objects={"Attention": Attention}, 
            compile=False
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        st.stop()
    
    try:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        st.stop()
    
    # Determine max sequence length
    # Check if tokenizer has maxlen attribute, otherwise use default or model input shape
    if hasattr(tokenizer, 'maxlen'):
        max_len = tokenizer.maxlen
    else:
        # Try to get from model input shape
        try:
            max_len = model.input_shape[1]
        except:
            max_len = 100  # Default fallback
    
    return model, tokenizer, le, max_len

# Load resources
try:
    model, tokenizer, le, max_len = load_resources()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load resources: {str(e)}")
    st.stop()

# User input
user_input = st.text_area("✍️ Enter your text here:", height=100)

if st.button("🔍 Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        try:
            # Preprocess input
            X_input_seq = tokenizer.texts_to_sequences([user_input])
            X_input_pad = pad_sequences(X_input_seq, maxlen=max_len, padding='pre')

            # Predict probabilities
            with st.spinner("Analyzing emotion..."):
                y_pred = model.predict(X_input_pad, verbose=0)
            
            predicted_index = np.argmax(y_pred, axis=1)
            predicted_emotion = le.inverse_transform(predicted_index)[0]

            # Show results
            st.markdown("---")
            st.subheader("📌 Prediction Result")
            st.success(f"**Predicted Emotion:** {predicted_emotion.upper()}")

            # All Probabilities graph
            prob_dict = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(y_pred[0])}
            prob_df = pd.DataFrame(prob_dict.items(), columns=["Emotion", "Probability"])
            prob_df = prob_df.sort_values(by="Probability", ascending=False)

            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(prob_df.set_index("Emotion"))
            
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Please check if the model files are compatible and properly saved.")