import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# attention mechanism
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
        self.u = self.add_weight(shape=(input_shape[-1],1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector


# Load model, tokenizer, and label encoder
try:
    model = load_model("emotion_bilstm_model.h5", custom_objects={"Attention": Attention}, compile=False)
except IndexError:
    # Fix for TensorFlow deserialization issue
    tf.keras.utils.get_custom_objects().clear()
    model = tf.keras.models.load_model(
        "emotion_lstm_model.h5",
        custom_objects={"Attention": Attention},
        compile=False,
        safe_mode=False
    )

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Streamlit UI

st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")
st.title("🎭 Text Emotion Classifier")
st.write("Enter any text below and let the model predict the **emotion**.")

# User input
user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess input
        X_input_seq = tokenizer.texts_to_sequences([user_input])
        X_input_pad = pad_sequences(X_input_seq, maxlen=tokenizer.maxlen if hasattr(tokenizer,'maxlen') else 100, padding='pre')

        # Predict probabilities
        y_pred = model.predict(X_input_pad)
        predicted_index = np.argmax(y_pred, axis=1) #argmax=shows highest porb.
        predicted_emotion = le.inverse_transform(predicted_index)[0]

        
        # Show results
        st.subheader("📌 Prediction")
        st.success(f"Predicted Emotion: **{predicted_emotion}**")

        #  All Probabilities graph
        prob_dict = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(y_pred[0])}
        prob_df = pd.DataFrame(prob_dict.items(), columns=["Emotion", "Probability"])
        prob_df = prob_df.sort_values(by="Probability", ascending=False)

        st.subheader("🔮 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Emotion"))
