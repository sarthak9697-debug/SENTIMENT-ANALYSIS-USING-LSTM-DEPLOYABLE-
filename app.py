import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
import re
import string
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

SAMPLE_TEXTS = {
    "Positive — strong performance": "Donald Trump performed really well in the debate",
    "Positive — supportive": "I really liked the candidate's answer tonight",
    "Negative — criticism": "That was a terrible answer and very disappointing",
    "Negative — poor performance": "The candidate performed badly and avoided the question",
    "Neutral — factual": "The debate started at nine tonight",
    "Neutral — general": "The candidates discussed several policy topics",
}

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl was not found. Run the deployment cell in the notebook first.")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = model_from_json(bundle["model_json"])
    model.set_weights(bundle["model_weights"])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model, bundle

def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def text_to_model_input(text, bundle):
    cleaned = clean_text(text, set(bundle["stop_words"]))
    word_index = bundle["tokenizer"]["word_index"]
    oov_token = bundle["tokenizer"].get("oov_token")
    oov_index = word_index.get(oov_token, 0) if oov_token else 0

    sequence = []
    for word in cleaned.split():
        sequence.append(word_index.get(word, oov_index))

    return pad_sequences(
        [sequence],
        maxlen=int(bundle["max_length_sequence"])
    ), cleaned

def predict_sentiment(text, model, bundle):
    x, cleaned = text_to_model_input(text, bundle)
    probabilities = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = bundle["label_classes"][index]
    confidence = float(probabilities[index])
    return label, confidence, probabilities, cleaned

st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis using LSTM")
st.caption("Recruiter-friendly demo of the LSTM text-classification model from the assignment notebook.")

with st.expander("About this project", expanded=True):
    st.write(
        "This application cleans tweet text, converts words into padded numerical sequences, "
        "and uses the trained LSTM model to classify the text as Negative, Neutral, or Positive."
    )
    st.write("Model architecture: Embedding → SpatialDropout1D → LSTM → Dense(3, softmax).")
    st.write("Maximum sequence length: 50 tokens | Vocabulary limit: 5,000 words.")

try:
    model, bundle = load_bundle()
except Exception as e:
    st.error(str(e))
    st.stop()

choice = st.selectbox("Try a built-in example", ["Custom input"] + list(SAMPLE_TEXTS.keys()))

if choice == "Custom input":
    text = st.text_area(
        "Enter a tweet or short sentence",
        value="",
        height=130,
        placeholder="Example: I think the candidate gave a strong answer tonight."
    )
else:
    text = st.text_area("Input text", value=SAMPLE_TEXTS[choice], height=130)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label, confidence, probabilities, cleaned = predict_sentiment(text, model, bundle)

        st.subheader("Prediction")
        if label.lower() == "positive":
            st.success(f"Positive — {confidence:.1%} confidence")
        elif label.lower() == "negative":
            st.error(f"Negative — {confidence:.1%} confidence")
        else:
            st.info(f"Neutral — {confidence:.1%} confidence")

        st.write("**Model confidence by class**")
        for cls, prob in zip(bundle["label_classes"], probabilities):
            st.progress(float(prob), text=f"{cls}: {prob:.1%}")

        with st.expander("See preprocessing"):
            st.write("Cleaned text:", cleaned if cleaned else "(empty after cleaning)")
            st.write(f"Padded sequence length: {bundle['max_length_sequence']}")

st.divider()
st.caption("Built for demonstration and interview/recruiter testing. Predictions depend on the training data and model quality.")
