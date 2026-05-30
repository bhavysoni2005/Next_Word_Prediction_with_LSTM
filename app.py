import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


st.write("Current Directory:", os.getcwd())
st.write("Files Available:", os.listdir())
# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    layout="wide"
)

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource
def load_resources():
    try:
        st.info(f"TensorFlow Version: {tf.__version__}")

        model = load_model(
            "lstm_model.h5",
            compile=False
        )

        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)

        # Reverse lookup dictionary
        index_to_word = {
            index: word
            for word, index in tokenizer.word_index.items()
        }

        return model, tokenizer, max_len, index_to_word

    except Exception as e:
        st.error("❌ Failed to load model resources.")
        st.exception(e)
        st.stop()


model, tokenizer, max_len, index_to_word = load_resources()

# ------------------------------
# Prediction Functions
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    return index_to_word.get(predicted_index, "")


def generate_text_advanced(
    seed_text,
    n_words,
    temperature=0.7,
    diversity_penalty=0.3
):
    generated_words = seed_text.split()
    current_text = seed_text

    prevent_repetition = 2

    for _ in range(n_words):

        seq = tokenizer.texts_to_sequences([current_text])[0]

        seq = pad_sequences(
            [seq],
            maxlen=max_len - 1,
            padding="pre"
        )

        preds = model.predict(seq, verbose=0)[0].copy()

        # Diversity penalty
        if diversity_penalty > 0:
            for recent_word in generated_words[-prevent_repetition:]:

                if recent_word in tokenizer.word_index:

                    idx = tokenizer.word_index[recent_word]

                    if idx < len(preds):
                        preds[idx] *= (1 - diversity_penalty)

        # Temperature scaling
        if temperature != 1.0:

            preds = np.asarray(preds).astype("float64")

            preds = np.log(preds + 1e-10) / temperature

            exp_preds = np.exp(preds)

            preds = exp_preds / np.sum(exp_preds)

            pred_index = np.random.choice(
                len(preds),
                p=preds
            )

        else:
            pred_index = np.argmax(preds)

        next_word = index_to_word.get(pred_index, "")

        if next_word == "":
            break

        generated_words.append(next_word)

        current_text += " " + next_word

    return " ".join(generated_words)

# ------------------------------
# UI
# ------------------------------
st.title("🧠 Next Word Prediction with LSTM")

st.write(
    "Generate creative text sequences using an LSTM neural network trained on quotes."
)

with st.expander("❓ How to Use This App", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📝 What is this?")

        st.write(
            """
            This app uses an LSTM neural network to generate
            text continuations from a seed phrase.
            """
        )

        st.markdown("### 🎯 How to Use")

        st.write(
            """
            1. Enter a starting phrase.
            2. Adjust generation settings.
            3. Click Generate.
            """
        )

    with col2:

        st.markdown("### ⚙️ Parameter Guide")

        st.write(
            """
            • Temperature controls creativity.

            • Low temperature = predictable output.

            • High temperature = more randomness.

            • Diversity penalty reduces repetition.
            """
        )

st.subheader("Generate Text Sequence")

col1, col2, col3 = st.columns(3)

with col1:
    gen_input = st.text_input(
        "💬 Seed Text",
        value="life is"
    )

with col2:
    n_words = st.slider(
        "Number of words",
        5,
        50,
        15
    )

with col3:
    temperature = st.slider(
        "Temperature",
        0.1,
        2.0,
        0.7,
        0.1
    )

diversity_penalty = st.slider(
    "Repetition Prevention",
    0.0,
    1.0,
    0.3,
    0.1
)

if st.button("🚀 Generate Text"):

    if not gen_input.strip():
        st.warning("Please enter seed text.")

    else:

        with st.spinner("Generating..."):

            generated_text = generate_text_advanced(
                gen_input,
                n_words,
                temperature,
                diversity_penalty
            )

        st.success("Generated Successfully!")

        st.text_area(
            "Generated Text",
            generated_text,
            height=150
        )

# ------------------------------
# Model Info
# ------------------------------
with st.expander("📊 Model Information"):

    st.write(
        """
        • Architecture: LSTM

        • Vocabulary Size: 10,000

        • Max Sequence Length: 48

        • Training Dataset: Quotes Dataset
        """
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h4>Bhavy Soni</h4>
        <p>LSTM Neural Network | Streamlit | TensorFlow</p>

        <p>
            <a href="https://github.com/bhavysoni2005">
                GitHub
            </a>
            |
            <a href="https://www.linkedin.com/in/bhavy-soni-b3746b316">
                LinkedIn
            </a>
        </p>

        <p>© 2026 Next Word Prediction AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
```
