import os
import warnings
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Created a chunk of size")

# Load environment variables
load_dotenv()

# Constants for file paths
TAGGED_DESC_PATH = "datasets/tagged_descriptions.txt"
EMBEDDINGS_PATH = "datasets/chroma_db"

# Variables for caching query results
current_query = None
current_recommendations = None
embedding_model = OpenAIEmbeddings()

# ---- Helper Functions ----


def add_thumbnails(books: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a large thumbnail URL to the books DataFrame. If a thumbnail is missing,
    a placeholder image is used.
    """
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isnull(),
        "resources/cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books


def create_and_save_embeddings(books: pd.DataFrame) -> Chroma:
    """
    Creates and saves embeddings for book descriptions if no embeddings exist.
    Embeddings are stored in a Chroma vector database for efficient similarity search.
    """
    if not os.path.exists(TAGGED_DESC_PATH):
        if not os.path.exists(TAGGED_DESC_PATH.split("/")[0]):
            os.mkdir(TAGGED_DESC_PATH.split("/")[0])

        # Combine ISBN and description into a tagged description
        books_copy = books.copy(deep=True)
        books_copy.loc[:, "tagged_description"] = (
            books_copy[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
        )
        books_copy["tagged_description"].to_csv(
            TAGGED_DESC_PATH, sep="\n", index=False, header=False
        )

    # Load raw documents and split into chunks
    raw_documents = TextLoader(TAGGED_DESC_PATH).load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Initialize Chroma DB to store embeddings
    db_books = Chroma(
        collection_name="book_descriptions",
        embedding_function=embedding_model,
        persist_directory=EMBEDDINGS_PATH,
    )

    # Create embeddings in batches and add to the database
    print(
        "No saved Vector DB found for books, so creating a new DB. It might take some time."
    )
    batch_size = 50
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        db_books.add_documents(batch)
    return db_books


def retrieve_semantic_recommendations(query: str, top_k: int = 50) -> pd.DataFrame:
    """
    Retrieves book recommendations based on semantic similarity to the query using
    the Chroma database.
    """
    recommendations = db_books.similarity_search(query=query, k=top_k)

    # Extract ISBNs from recommendations
    books_list = [
        int(rec.page_content.strip('"').split()[0].strip()) for rec in recommendations
    ]

    # Filter books DataFrame by ISBNs
    book_recs = books[books["isbn13"].isin(books_list)]
    return book_recs


def filter_recommendations(
    recommendations: pd.DataFrame, category: str, tone: str, fetch_number: int = 16
) -> pd.DataFrame:
    """
    Filters the book recommendations by category and tone.
    """
    # Filter by category
    if category != "All":
        recommendations = recommendations[
            recommendations["simple_categories"] == category
        ].head(fetch_number)
    else:
        recommendations = recommendations.head(fetch_number)

    # Sort by tone-based emotion
    tone_column_map = {
        "Happy": "emotion_joy",
        "Surprising": "emotion_surprise",
        "Angry": "emotion_anger",
        "Suspenseful": "emotion_fear",
        "Sad": "emotion_sadness",
    }

    if tone in tone_column_map:
        recommendations.sort_values(
            by=tone_column_map[tone], ascending=False, inplace=True
        )

    return recommendations


def recommend_books(query: str, category: str, tone: str):
    """
    Generates book recommendations based on a user's query, category, and tone preferences.
    """
    global current_query, current_recommendations

    # If the query is the same as the previous one, use cached recommendations
    if current_query is not None and current_query == query:
        recommendations = current_recommendations.copy(deep=True)
    else:
        # Retrieve new recommendations
        recommendations = retrieve_semantic_recommendations(query=query, top_k=50)
        current_query = query
        current_recommendations = recommendations.copy(deep=True)

    # Filter recommendations by category and tone
    recommendations = filter_recommendations(recommendations, category, tone)

    # Format and prepare the results for display
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format authors list
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create caption for the book
        caption_text = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption_text))
    return results


# ---- Main Section ----

if __name__ == "__main__":
    # Load books data and add thumbnails
    books = pd.read_csv("datasets/books_with_categories_and_emotions.csv")
    books = add_thumbnails(books)

    # Define available categories and tones
    categories = ["All"] + sorted(books["simple_categories"].unique())
    tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    db_books = None
    try:
        # Attempt to load the existing embeddings database
        if not os.path.exists(EMBEDDINGS_PATH):
            raise Exception("No saved embeddings found")

        db_books = Chroma(
            collection_name="book_descriptions",
            embedding_function=embedding_model,
            persist_directory=EMBEDDINGS_PATH,
        )

        # Validate the number of documents in the database
        if db_books._collection.count() != len(books):
            raise Exception("Invalid embeddings loaded")
    except Exception as e:
        # Recreate embeddings if the database is invalid or missing
        print(f"{str(e)}, Recreating embeddings for books")
        db_books = create_and_save_embeddings(books)

    # ---- Gradio Interface ----

    # Initialize Gradio dashboard
    with gr.Blocks(theme=gr.themes.Base()) as dashboard:
        # Main title
        gr.Markdown("# Semantic Book Recommender")

        with gr.Row():
            # Textbox for user input
            user_query = gr.Textbox(
                label="Please enter a description of a book:",
                placeholder="e.g., A book to teach children about nature",
                elem_id="query-textbox",
            )

            # Category dropdown for filtering
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Select a category:",
                value="All",
                elem_id="category-dropdown",
            )

            # Tone dropdown for emotional tone filtering
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Select an emotional tone:",
                value="All",
                elem_id="tone-dropdown",
            )

            # Styled button
            submit_button = gr.Button(
                value="Find Book Recommendations",
                elem_id="submit-button",
            )

        gr.HTML("<hr>")

        # Recommendations display area
        gr.Markdown("## Recommendations")
        output = gr.Gallery(
            label="Recommended books", columns=8, rows=2, elem_id="recom-tbl"
        )

        # Connect button click with recommendation function
        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output,
        )

    dashboard.title = "Book Recommender"
    dashboard.head = """
        <title>Book Recommender</title>
        <link rel="icon" type="image/png" href="https://raw.githubusercontent.com/lokeshagg13/Book-Recommender/refs/heads/main/resources/favicon_io/favicon-32x32.png">
    """
    
    # Styling the components for a more polished look
    dashboard.css = """
        /* Custom CSS for a white theme with modern styling */
        .gradio-container {
            background-color: #fff; /* White background */
            padding: 20px;
            border-radius: 10px;
        }

        .gradio-container h1, h2 {
            color: black !important;
        }
        
        
        /* Textbox and dropdown styling */
        #query-textbox, #category-dropdown, #tone-dropdown {
            font-size: 14px;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
        }
        
        span[data-testid="block-info"] {
            color: black !important;
        }
        
        /* Button styling */
        #submit-button {
            background-color: #4CAF50; /* Green background */
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        #submit-button:hover {
            background-color: #45a049; /* Darker green on hover */
            cursor: pointer;
        }

        /* Gallery styling */
        .gr-gallery-item {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .gr-gallery-item:hover {
            transform: scale(1.05); /* Slight zoom effect on hover */
        }
        
        #recom-tbl caption {
            white-space: normal !important;
            text-align: center;
            font-size: 14px;
            padding: 5px;
            margin: -10px;
        }   
        """

    # Launch Gradio interface
    dashboard.launch(share=True)
