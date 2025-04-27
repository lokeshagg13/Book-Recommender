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

---

## ✨ How It Works

- **Load** book descriptions into documents
- **Split** text into manageable chunks
- **Embed** documents using OpenAI’s `text-embedding-ada-002` model
- **Store** embeddings in **Chroma** (persisted locally in SQLite and binary files)
- **Query** the database semantically to find similar books

The embeddings and metadata are stored in the `datasets/chroma_db/` directory.

---

## 🔮 Upcoming Features

- **Text Classification**:  
  Organizing and grouping books into more meaningful categories using ML models.

- **Sentiment Analysis**:  
  Analyzing book descriptions to classify sentiment and improve recommendation relevance.

---

## 🛠️ Tech Stack

- Python (Jupyter Notebooks)
- LangChain
- OpenAI API
- ChromaDB
- dotenv
- Pandas, NumPy
- Matplotlib, Seaborn (optional)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute!
