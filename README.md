# 📚 Book Recommender Application

Welcome to the **Book Recommender**!  
An intelligent system built using **LangChain**, **OpenAI embeddings**, and a dataset of 7K+ books to help users find books similar to their interests.  
This project uses **semantic search** powered by **vector databases** for smarter, meaning-based recommendations.

---

## 📂 Dataset Information

Dataset sourced from Kaggle: [7k+ Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

**Main columns:**

| Column Name      | Description                                          |
|------------------|------------------------------------------------------|
| `isbn13`          | ISBN 13 Identifier                                   |
| `isbn10`          | ISBN 10 Identifier                                   |
| `title`           | Title of the book                                    |
| `subtitle`        | Subtitle of the book                                 |
| `authors`         | Authors (separated by `;`)                           |
| `categories`      | Categories/Genres (separated by `;`)                 |
| `thumbnail`       | URL of the book thumbnail                            |
| `description`     | Book description                                     |
| `published_year`  | Year the book was published                          |
| `average_rating`  | Average user rating (where available)                |

---

## 🔥 Features Implemented

- **Data Exploration**: Cleaning, understanding, and preparing book metadata.
- **Vector Search Engine**:
  - Uses **LangChain** for document loading, text splitting, and embedding generation.
  - **OpenAI's Embedding Models** used to embed book descriptions into vector space.
  - **ChromaDB** as a lightweight, fast vector database for storage and retrieval.
- **Semantic Search**: Find books not just by keywords, but based on meaning!
- **Batch Processing**: Batching documents for efficient embedding within OpenAI’s rate limits.
- **Text Classification using Zero-shot Learning**:
  - Broadly categorizing books into higher-level groups (Fiction, Nonfiction, Children's categories).
  - Using **Hugging Face Transformers** (e.g., `facebook/bart-large-mnli`) without additional model training.
- **Sentiment and Emotion Analysis**:
  - Predicting emotional tone (anger, joy, sadness, etc.) from book descriptions.
  - Fine-tuned transformer models used (e.g., `j-hartmann/emotion-english-distilroberta-base`).
  - Allows filtering books based on emotional tone for better personalized recommendations.
- **Environment Management**:
  - API keys and other secrets managed securely using `.env` files.
  - Requires `OPENAI_API_KEY` to be set.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/book-recommender.git
cd book-recommender
```

### 2. Set up a virtual environment (recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure you have Python 3.10+ installed.

### 4. Setup your API Key

Create a `.env` file in the project root directory and add your **OpenAI API Key**:

```
OPENAI_API_KEY=your_openai_api_key_here
```

You can find or create your key at [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).

### 5. Additional Setup for Hugging Face and Mac MPS (Apple Silicon) Support

If you plan to use **Hugging Face Transformers** for **zero-shot classification** and want to enable hardware acceleration on your Mac (using **MPS** — Metal Performance Shaders), you need to install **PyTorch** correctly.

#### (I) Install PyTorch with MPS support (for macOS)

Follow the official installation instructions [here](https://pytorch.org/get-started/locally/), or run the following for macOS:

```bash
# Install PyTorch for macOS with MPS backend
pip install torch torchvision torchaudio
```

> **Note**: MPS support is available on macOS 12.3+ with Apple Silicon (M1, M2, M3 chips) or newer Intel Macs.

---

#### (II) Check if MPS is available

After installing, you can check MPS support with this snippet:

```python
import torch

if torch.backends.mps.is_available():
    print("✅ MPS is available. Using hardware acceleration!")
else:
    print("⚠️ MPS is not available. Running on CPU.")
```

#### (III) Why is PyTorch Required?

The Hugging Face `transformers` library needs either **PyTorch**, **TensorFlow**, or **Flax** as a backend to load models like `facebook/bart-large-mnli`.  
Without a deep learning backend, you will only be able to use tokenizers and configs — not the actual models.

#### (IV) Additional Requirements for Hugging Face

Make sure you have installed:

```bash
pip install transformers
```

If not already present, install `huggingface_hub` and other related libraries too:

```bash
pip install huggingface_hub
```

---

## ✨ How It Works

- **Load** book descriptions into documents
- **Split** text into manageable chunks
- **Embed** documents using OpenAI’s `text-embedding-ada-002` model
- **Store** embeddings in **Chroma** (persisted locally in SQLite and binary files)
- **Query** the database semantically to find similar books
- **Classify** books into higher-level groups to support better filtering
- **Analyze** emotional tones of descriptions for sentiment-based recommendation enhancements

The embeddings and metadata are stored in the `datasets/chroma_db/` directory.

---

## 🔮 Upcoming Features

- **Gradle Dashboard**:  
  An interactive dashboard to visually explore books, categories, and search recommendations.

---

## 🛠️ Tech Stack

- Python (Jupyter Notebooks)
- LangChain
- OpenAI API
- Hugging Face Transformers
- ChromaDB
- dotenv
- Pandas, NumPy
- Matplotlib, Seaborn (optional)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute!
