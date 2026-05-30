# 🧠 Next Word Prediction using LSTM

A Deep Learning based Natural Language Processing (NLP) project that predicts the next word in a sentence using an LSTM (Long Short-Term Memory) neural network.

This project demonstrates how sequential text data can be processed and trained using deep learning techniques to generate meaningful next-word predictions, similar to autocomplete systems used in modern keyboards and search engines.

---

# 🚀 Features

* Text preprocessing and tokenization
* Sequence generation using NLP techniques
* LSTM-based deep learning model
* Predicts the next probable word from input text
* Built using TensorFlow/Keras
* Easy to train with custom datasets

---

# 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* NLP
* Jupyter Notebook

---

# 📂 Project Structure

```bash
Next_Word_Prediction_with_LSTM/
│
├── Next_Word_Prediction.ipynb   # Main training notebook
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies (optional)
└── dataset.txt                  # Training text dataset
```

---

# 📊 How It Works

## 1. Data Preprocessing

* Convert text into lowercase
* Remove unnecessary characters
* Tokenize words
* Create input sequences

## 2. Sequence Generation

Example:

Input:

```text
I love deep
```

Target:

```text
learning
```

The model learns word patterns from sequences.

---

## 3. Model Architecture

The project uses:

* Embedding Layer
* LSTM Layer
* Dense Output Layer

LSTM helps retain long-term dependencies in text sequences, making it effective for language modeling tasks.

---

# 🧪 Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/bhavysoni2005/Next_Word_Prediction_with_LSTM.git
```

## Navigate to Project Folder

```bash
cd Next_Word_Prediction_with_LSTM
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Run all cells in:

```bash
Next_Word_Prediction.ipynb
```

---

# 💡 Example Prediction

### Input:

```text
Machine learning is
```

### Predicted Output:

```text
fun
```

Tiny robot brain predicts text. Humanity applauds and immediately uses it to write lazy Instagram captions. Evolution is fascinating.

---

# 📈 Future Improvements

* Add GRU and Bidirectional LSTM models
* Improve prediction accuracy with larger datasets
* Deploy using Flask or Streamlit
* Add beam search text generation
* Build real-time autocomplete web app

---

# 🎯 Applications

* Text Autocomplete
* Smart Keyboards
* Chatbots
* AI Writing Assistants
* Search Engines
* Text Generation Systems

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Make changes
4. Submit a pull request

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

**Bhavy Soni**

* GitHub: https://github.com/bhavysoni2005

---

# ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.
It helps more people discover the project and feeds the developer ego just enough to continue debugging TensorFlow errors.
