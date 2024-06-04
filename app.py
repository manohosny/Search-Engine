import time

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pyterrier as pt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
import tensorflow_hub as hub

elmo = hub.load("https://tfhub.dev/google/elmo/3")

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def Stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)


def clean(text):
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=\@]", " ", text)  # remove special characters
    text = re.sub(r'\t', ' ', text)  # remove tabs
    text = re.sub(r'\n', ' ', text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space
    text = text.strip()
    return text


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if
                       word.lower() not in stop_words]  # Lower is used to normalize al the words make them in lower case
    return ' '.join(filtered_tokens)


# we need to process the query also as we did for documents
def preprocess(sentence):
    sentence = clean(sentence)
    sentence = remove_stopwords(sentence)
    sentence = Stem_text(sentence)
    return sentence


import zipfile

zip_file_name = 'cisi.zip'
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall('cisi_dataset')


def load_cisi_dataset(data_dir):
    documents_path = os.path.join(data_dir, 'CISI.ALL')
    queries_path = os.path.join(data_dir, 'CISI.QRY')
    qrels_path = os.path.join(data_dir, 'CISI.REL')

    documents_df = read_documents(documents_path)
    queries_df = read_queries(queries_path)
    qrels_df = read_qrels(qrels_path)
    return documents_df, queries_df, qrels_df


# Read documents from CISI.ALL file
def read_documents(documents_path):
    with open(documents_path, 'r') as file:
        lines = file.readlines()
    documents = []
    current_document = None
    for line in lines:
        if line.startswith('.I'):
            if current_document is not None:
                current_document['Text'] = current_document['Text'].split('\t')[
                    0].strip()  # Remove anything after the first tab
                documents.append(current_document)
            current_document = {'ID': line.strip().split()[1], 'Text': ''}
        elif line.startswith('.T'):
            continue
        elif line.startswith('.A') or line.startswith('.B') or line.startswith('.W') or line.startswith('.X'):
            continue
        else:
            current_document['Text'] += line.strip() + ' '

    # Append the last document
    if current_document is not None:
        current_document['Text'] = current_document['Text'].split('\t')[
            0].strip()  # Remove anything after the first tab
        documents.append(current_document)
    documents_df = pd.DataFrame(documents)
    return documents_df


# Read queries from CISI.QRY file
def read_queries(queries_path):
    with open(queries_path, 'r') as file:
        lines = file.readlines()
    query_texts = []
    query_ids = []
    current_query_id = None
    current_query_text = []
    for line in lines:
        if line.startswith('.I'):
            if current_query_id is not None:
                query_texts.append(' '.join(current_query_text))
                current_query_text = []
            current_query_id = line.strip().split()[1]
            query_ids.append(current_query_id)
        elif line.startswith('.W'):
            continue
        elif line.startswith('.X'):
            break
        else:
            current_query_text.append(line.strip())
    # Append the last query
    query_texts.append(' '.join(current_query_text))
    queries_df = pd.DataFrame({
        'qid': query_ids,
        'raw_query': query_texts})
    return queries_df


# Read qrels from CISI.REL file
def read_qrels(qrels_path):
    qrels_df = pd.read_csv(qrels_path, sep='\s+', names=['qid', 'Q0', 'docno', 'label'])
    return qrels_df


data_dir = 'cisi_dataset'
documents_df, queries_df, qrels_df = load_cisi_dataset(data_dir)
qrels_df = qrels_df.drop(columns=['Q0'])
documents_df["docno"] = documents_df["ID"].astype(str)
queries_df["qid"] = queries_df["qid"].astype(str)
documents_df['processed_text'] = documents_df['Text'].apply(preprocess)
queries_df["query"] = queries_df["raw_query"].apply(preprocess)

if not pt.started():
    # In this lab, we need to specify that we start PyTerrier with PRF enabled
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)
# index the text, record the docnos as metadata
index_ref = indexer.index(documents_df["processed_text"], documents_df["docno"])

index = pt.IndexFactory.of(index_ref)

TF_IDF = pt.BatchRetrieve(index, wmodel="TF_IDF", num_results=10)

def preprocess_queries(sentence):
  sentence = clean(sentence)
  sentence = remove_stopwords(sentence)
  return sentence
def expand_query(input_query):
  input_query = preprocess(str(input_query))
  search_results = TF_IDF.search(input_query)
  merged_results = pd.merge(search_results, documents_df, on='docno', how='inner')
  text_list = merged_results["Text"][:2].to_list()
  processed_list = np.array([preprocess_queries(text) for text in text_list])
  document1 = preprocess_queries(processed_list[0])
  document2 = preprocess_queries(processed_list[1])
  input_query = preprocess_queries(input_query)

  embeddings_document1 = elmo.signatures["default"](tf.constant([document1]))["elmo"]
  embeddings_document2 = elmo.signatures["default"](tf.constant([document2]))["elmo"]
  embeddings_input_query = elmo.signatures["default"](tf.constant([input_query]))["elmo"]

  # Average the embeddings over the sequence length dimension
  embeddings_document1 = tf.reduce_mean(embeddings_document1, axis=1)
  embeddings_document2 = tf.reduce_mean(embeddings_document2, axis=1)
  embeddings_input_query = tf.reduce_mean(embeddings_input_query, axis=1)

  # Now you can normalize the 2D tensors
  normalized_embeddings_document1 = normalize(embeddings_document1.numpy(), axis=1)
  normalized_embeddings_document2 = normalize(embeddings_document2.numpy(), axis=1)
  normalized_embeddings_input_query = normalize(embeddings_input_query.numpy(), axis=1)

  similarity_scores_document1 = cosine_similarity(normalized_embeddings_input_query, normalized_embeddings_document1)
  similarity_scores_document2 = cosine_similarity(normalized_embeddings_input_query, normalized_embeddings_document2)

  max_similarity_indices_document1 = np.argmax(similarity_scores_document1, axis=1)
  max_similarity_indices_document2 = np.argmax(similarity_scores_document2, axis=1)
  index = 1
  while index < len(max_similarity_indices_document1):
      if max_similarity_indices_document1[index] == max_similarity_indices_document1[index - 1]:
          max_similarity_indices_document1 = np.delete(max_similarity_indices_document1, index)
      else:
          index += 1
  while index < len(max_similarity_indices_document2):
      if max_similarity_indices_document2[index] == max_similarity_indices_document2[index - 1]:
          max_similarity_indices_document2 = np.delete(max_similarity_indices_document2, index)
      else:
          index += 1

  for idx in max_similarity_indices_document1:
      input_query += " " + processed_list[0].split()[idx]

  for idx in max_similarity_indices_document2:
      input_query += " " + processed_list[1].split()[idx]

  input_query = preprocess(input_query)
  query_words = input_query.split()
  distinct_words = []
  for word in query_words:
      if word not in distinct_words:
          distinct_words.append(word)

  input_query = ' '.join(distinct_words)
  print(input_query)
  return input_query

def precision_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / len(predicted)

def recall_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / len(actual)

def average_precision(actual, predicted):
    precisions = []
    for i, p in enumerate(predicted):
        if p in actual:
            precisions.append(precision_at_k(actual, predicted, i+1))
    if len(precisions) == 0:
        return 0
    return np.mean(precisions)

def mean_average_precision(actuals, predicteds):
    return np.mean([average_precision(a, p) for a, p in zip(actuals, predicteds)])
app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.get_json()['query']
        res = expand_query(query)
        start_time = time.time()
        res = TF_IDF.search(res)
        precisions = [precision_at_k(a, p, 10) for a, p in zip(qrels_df, res)]
        recalls = [recall_at_k(a, p, 10) for a, p in zip(qrels_df, res)]
        map_score = mean_average_precision(qrels_df, res)
        print("MAP: ", map_score)
        print("P@10: ", np.mean(precisions))
        print("R@10: ", np.mean(recalls))
        end_time = time.time()
        retrieval_time = end_time - start_time
        result_merged = res.merge(documents_df, on="docno",how = "inner")[["score", "Text", "docno"]]
        result_merged = result_merged.sort_values(by="score", ascending=False)
        if 'score' in result_merged.columns:
            res = result_merged.rename(columns={'docno': 'doc_id', 'Text':'text','score': 'score'})
            res['retrieval_time'] = retrieval_time 
            return res.to_json(orient='records')
        else:
            return jsonify({"error": "Score not found in the response"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
