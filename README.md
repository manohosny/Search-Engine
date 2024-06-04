# Search Engine README

## Overview

This project is a web-based search engine built using Flask, PyTerrier, and various NLP techniques. It processes, indexes, and retrieves documents from the CISI dataset. Additionally, it uses ELMo embeddings for query expansion and calculates various information retrieval metrics.

## Features

- **Document Processing**: Cleans and preprocesses text data by removing special characters, stop words, and applying stemming.
- **Query Expansion**: Expands user queries using ELMo embeddings to improve search results.
- **Information Retrieval**: Uses PyTerrier's TF_IDF model for retrieving relevant documents.
- **Evaluation Metrics**: Calculates precision, recall, and mean average precision (MAP) for the search results.

## Requirements

- Python 3.x
- Flask
- pandas
- pyterrier
- nltk
- scikit-learn
- tensorflow
- tensorflow_hub

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/manohosny/search-engine.git
   cd search-engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. **Download ELMo embeddings**:
   ```python
   import tensorflow_hub as hub
   elmo = hub.load("https://tfhub.dev/google/elmo/3")
   ```

5. **Prepare the dataset**:
   Ensure the CISI dataset (`cisi.zip`) is in the project directory. The dataset will be extracted and processed automatically.

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://127.0.0.1:5000/`.

## Endpoints

### `POST /search`
- **Description**: Searches for documents based on the provided query.
- **Request**: JSON object with a `query` field.
  ```json
  {
      "query": "sample search query"
  }
  ```
- **Response**: JSON array of search results with document ID, text, score, and retrieval time.
  ```json
  [
      {
          "doc_id": "1",
          "text": "Document content here...",
          "score": 0.85,
          "retrieval_time": 0.123
      },
      ...
  ]
  ```

### `GET /`
- **Description**: Renders the home page.
- **Response**: HTML page.

## Code Structure

- **app.py**: Main Flask application file containing routes and search logic.
- **templates/**: Directory for HTML templates.
- **static/**: Directory for static files (CSS, JS, images).
- **cisi_dataset/**: Directory where the CISI dataset will be extracted.

## Key Functions

### Text Processing
- `clean(text)`: Removes special characters, tabs, line jumps, and extra white spaces from the text.
- `remove_stopwords(text)`: Removes stop words from the text.
- `Stem_text(text)`: Applies stemming to the text.
- `preprocess(sentence)`: Combines cleaning, stop words removal, and stemming.

### Dataset Handling
- `load_cisi_dataset(data_dir)`: Loads the CISI dataset.
- `read_documents(documents_path)`: Reads documents from the CISI.ALL file.
- `read_queries(queries_path)`: Reads queries from the CISI.QRY file.
- `read_qrels(qrels_path)`: Reads relevance judgments from the CISI.REL file.

### Query Expansion
- `expand_query(input_query)`: Expands the input query using ELMo embeddings.

### Evaluation Metrics
- `precision_at_k(actual, predicted, k)`: Calculates precision at K.
- `recall_at_k(actual, predicted, k)`: Calculates recall at K.
- `average_precision(actual, predicted)`: Calculates average precision.
- `mean_average_precision(actuals, predicteds)`: Calculates mean average precision.

## Notes

- Ensure TensorFlow and TensorFlow Hub are correctly installed to load ELMo embeddings.
- Modify `cisi.zip` path if it is located in a different directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
