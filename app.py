import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction functions
# ------------------------------
def predict_next_word(text):
    """Predict the most likely next word"""
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

def generate_text_advanced(seed_text, n_words, temperature=0.7, diversity_penalty=0.3):
    """Generate text with advanced strategies to prevent repetition"""
    generated_words = seed_text.split()
    current_text = seed_text
    prevent_repetition = 2
    
    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([current_text])[0]
        seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
        
        preds = model.predict(seq, verbose=0)[0].copy()
        
        # Apply diversity penalty to recent words
        if diversity_penalty > 0:
            for recent_word in generated_words[-prevent_repetition:]:
                if recent_word in tokenizer.word_index:
                    idx = tokenizer.word_index[recent_word]
                    if idx < len(preds):
                        preds[idx] = preds[idx] * (1 - diversity_penalty)
        
        # Apply temperature
        if temperature != 1.0:
            preds = np.power(preds, 1.0/temperature)
            preds = preds / (np.sum(preds) + 1e-10)
            pred_index = np.random.choice(len(preds), p=preds)
        else:
            pred_index = np.argmax(preds)
        
        # Get word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == pred_index:
                next_word = word
                break
        
        if next_word == "":
            break
        
        generated_words.append(next_word)
        current_text += " " + next_word
    
    return " ".join(generated_words)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="wide")

st.title("🧠 Next Word Prediction with LSTM")
st.write("Generate creative text sequences using an LSTM neural network trained on quotes.")

# Help Section
with st.expander("❓ How to Use This App", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 What is this?")
        st.write("""
        This app uses a **Long Short-Term Memory (LSTM)** neural network to generate 
        creative text continuations. It learns patterns from famous quotes to predict 
        the next words in a sequence.
        """)
        
        st.markdown("### 🎯 How to Use:")
        st.write("""
        1. **Enter Seed Text**: Start with any phrase (e.g., "life is", "the world")
        2. **Adjust Parameters**: Control how long and diverse the output is
        3. **Click Generate**: Watch the AI complete your text!
        """)
    
    with col2:
        st.markdown("### ⚙️ Parameter Guide:")
        st.write("""
        - **Seed Text**: Starting phrase for text generation
        - **Words to Generate**: How many words to add (5-50)
        - **Temperature**: 
          - Low (0.1): Conservative, repetitive
          - Medium (0.7): Balanced, recommended ✓
          - High (2.0): Creative, random
        - **Repetition Prevention**: Avoids repeating words (higher = less repetition)
        """)
    
    st.markdown("### 💡 Tips for Best Results:")
    tips = """
    - Use **3-5 words** as seed text for better context
    - Try temperature **0.7-0.9** for natural-sounding text
    - Use repetition prevention **0.3-0.5** to avoid word loops
    - Try different seed texts: "love is", "the best", "success is", etc.
    - Generate **10-20 words** for coherent sentences
    """
    st.write(tips)

st.subheader("Generate Text Sequence")

col1, col2, col3 = st.columns(3)
with col1:
    gen_input = st.text_input("💬 Seed text:", placeholder="Start with...", value="life is")
with col2:
    n_words = st.slider("Number of words to generate:", 5, 50, 15)
with col3:
    temperature = st.slider("Temperature (diversity):", 0.1, 2.0, 0.7, 0.1)

col4, col5 = st.columns(2)
with col4:
    diversity_penalty = st.slider("Repetition prevention:", 0.0, 1.0, 0.3, 0.1)
    
if st.button("Generate Text", key="btn_generate"):
    if gen_input.strip() == "":
        st.warning("Please enter some seed text.")
    else:
        with st.spinner("Generating text..."):
            generated_text = generate_text_advanced(
                gen_input, 
                n_words, 
                temperature=temperature,
                diversity_penalty=diversity_penalty
            )
        st.success("✅ Text generated!")
        st.text_area("Generated Text:", generated_text, height=100, disabled=True)

# Display model info
with st.expander("📊 Model Information"):
    st.write("""
    - **Architecture:** LSTM Neural Network
    - **Vocabulary Size:** 10,000 words
    - **Max Sequence Length:** 48
    - **Training Data:** Quote dataset
    - **Final Accuracy:** 60.32% (training), 50.68% (validation)
    """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

# Professional Footer with Personal Info
footer_content = """
<div style="text-align: center; padding: 20px;">
    <p style="font-size: 14px; margin: 5px 0;">
        <b>Developer:</b> Bhavy Soni
    </p>
    <p style="font-size: 12px; color: #888; margin: 8px 0;">
        🧠 LSTM Neural Network | Built with Streamlit | Powered by TensorFlow & Keras
    </p>
    <p style="font-size: 12px; margin: 10px 0;">
        <a href="https://github.com/bhavysoni2005" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <b>GitHub</b> 🐙
        </a> | 
        <a href="https://www.linkedin.com/in/bhavy-soni-b3746b316" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <b>LinkedIn</b> 💼
        </a>
    </p>
    <p style="font-size: 11px; color: #999; margin: 10px 0;">
        © 2026 Next Word Prediction AI | All Rights Reserved
    </p>
</div>
"""
st.markdown(footer_content, unsafe_allow_html=True)