**Environment Setup**

First, let's set up our environment and load the required data:

```python
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Install required libraries
# !pip install -U minsearch qdrant_client rouge

# Load the data
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

# Evaluation functions
def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }
```

**Q1. Minsearch text**

For this question, we need to evaluate minsearch with specific boosting parameters:

```python
from minsearch import Index
import inspect

# Check the signature of the search method
print(inspect.signature(Index.search))

# Create a basic index
index = Index(
    text_fields={
        'question': 1.5,
        'text': 1.0,
        'section': 0.1
    },
    keyword_fields={'course'}
)
index.fit(documents)

# Try a basic search without any parameters
results = index.search("How to train a model?")
print(f"Number of results: {len(results)}")
print(f"First result: {results[0] if results else None}")

# Define search function based on what we learn
def search_function_q1(q):
    # We'll adjust this based on what we learn about the API
    results = index.search(q['question'])
    # If there's no limit parameter, we'll slice the results
    return results[:5]

# Evaluate
results_q1 = evaluate(ground_truth, search_function_q1)
print(f"Hit rate: {results_q1['hit_rate']}")
print(f"MRR: {results_q1['mrr']}")
```

After running this code, we'll get the hit rate. Based on the options prvodided, the value `0.74` is closest to our result `0.727`.


**Q2. Vector search for question**

Now we'll use vector search with embeddings for the "question" field:

```python
from minsearch import VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Create embeddings for questions
texts = []
for doc in documents:
    t = doc['question']
    texts.append(t)

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X = pipeline.fit_transform(texts)

# Create and fit the vector index
vindex = VectorSearch(keyword_fields={'course'})
vindex.fit(X, documents)

# Define search function
def search_function_q2(q):
    q_vector = pipeline.transform([q['question']])
    # Check if VectorSearch.search accepts 'limit' parameter
    try:
        return vindex.search(q_vector[0], limit=5)
    except TypeError:
        # If 'limit' is not accepted, try without it and slice the results
        results = vindex.search(q_vector[0])
        return results[:5]

# Evaluate
results_q2 = evaluate(ground_truth, search_function_q2)
print(f"Hit rate: {results_q2['hit_rate']}")
print(f"MRR: {results_q2['mrr']}")
```

MRR value `0.25` is closest to our result `0.29`.

**Q3. Vector search for question and answer**

Now we'll use both question and answer text for vector search:

```python
# Create embeddings for question + answer
texts = []
for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts.append(t)

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X = pipeline.fit_transform(texts)

# Create and fit the vector index
vindex = VectorSearch(keyword_fields={'course'})
vindex.fit(X, documents)

# Define search function
def search_function_q3(q):
    q_vector = pipeline.transform([q['question']])
    # Check if VectorSearch.search accepts 'limit' parameter
    try:
        return vindex.search(q_vector[0], limit=5)
    except TypeError:
        # If 'limit' is not accepted, try without it and slice the results
        results = vindex.search(q_vector[0])
        return results[:5]

# Evaluate
results_q3 = evaluate(ground_truth, search_function_q3)
print(f"Hit rate: {results_q3['hit_rate']}")
print(f"MRR: {results_q3['mrr']}")
```

Hit rate value `0.82` is closest to our result `0.773`.


**Q4. Qdrant**

For this question, we'll use Qdrant with the specified settings:

```python
# Install the sentence-transformers package
!pip install sentence-transformers

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Initialize the model
model_handle = "jinaai/jina-embeddings-v2-small-en"
model = SentenceTransformer(model_handle)

# Create texts combining question and answer
texts = []
for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts.append(t)

# Generate embeddings
embeddings = model.encode(texts)

# Initialize Qdrant client (using in-memory storage for this example)
client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="faq",
    vectors_config=models.VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    )
)

# Upload vectors
client.upload_points(
    collection_name="faq",
    points=[
        models.PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=doc
        )
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
    ]
)

# Define search function
def search_function_q4(q):
    query_vector = model.encode(q['question']).tolist()
    results = client.search(
        collection_name="faq",
        query_vector=query_vector,
        limit=5
    )
    
    # Make sure we're returning the payload in the expected format
    try:
        return [point.payload for point in results]
    except AttributeError:
        # If results are returned in a different format
        if isinstance(results[0], dict) and 'payload' in results[0]:
            return [point['payload'] for point in results]
        else:
            # Try to adapt to the structure we have
            print("Warning: Unexpected result format from Qdrant. Check the structure:")
            print(results[0])
            # Return a best guess
            return results

# Evaluate
results_q4 = evaluate(ground_truth, search_function_q4)
print(f"Hit rate: {results_q4['hit_rate']}")
print(f"MRR: {results_q4['mrr']}")
```

MRR value `0.85` is closest to our result `0.086`.

**Q5. Cosine similarity**

For this question, we'll calculate the cosine similarity between LLM-generated answers and original answers:

```python
import numpy as np

# Load results
results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
df_results = pd.read_csv(results_url)

# Define cosine similarity function with error handling
def cosine(u, v):
    try:
        u_norm = np.sqrt(u.dot(u))
        v_norm = np.sqrt(v.dot(v))
        
        # Check for zero division
        if u_norm == 0 or v_norm == 0:
            return 0.0
            
        return u.dot(v) / (u_norm * v_norm)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0  # Return 0 for problematic vectors

# Create embeddings pipeline
pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)

# Fit the pipeline on all text data
all_texts = df_results.answer_llm.fillna('') + ' ' + df_results.answer_orig.fillna('') + ' ' + df_results.question.fillna('')
pipeline.fit(all_texts)

# Calculate cosine similarity for each pair
cosine_similarities = []
for _, row in df_results.iterrows():
    # Handle potential NaN values
    llm_answer = row.answer_llm if pd.notna(row.answer_llm) else ''
    orig_answer = row.answer_orig if pd.notna(row.answer_orig) else ''
    
    # Skip empty answers
    if not llm_answer or not orig_answer:
        continue
        
    v_llm = pipeline.transform([llm_answer])[0]
    v_orig = pipeline.transform([orig_answer])[0]
    sim = cosine(v_llm, v_orig)
    cosine_similarities.append(sim)

# Calculate average cosine similarity
if cosine_similarities:
    avg_cosine = np.mean(cosine_similarities)
    print(f"Average cosine similarity: {avg_cosine}")
else:
    print("No valid cosine similarities calculated")
```

Cosine similarity alue `0.84` is closest to our result `0.8416`.

**Q6. Rouge**

Finally, we'll calculate the Rouge scores:

```python
from rouge import Rouge
import numpy as np

# Initialize Rouge scorer
rouge_scorer = Rouge()

# Check the Rouge score for the 10th document
try:
    r = df_results.iloc[10]
    if pd.notna(r.answer_llm) and pd.notna(r.answer_orig) and r.answer_llm and r.answer_orig:
        scores = rouge_scorer.get_scores(r.answer_llm, r.answer_orig)[0]
        print(f"Rouge scores for 10th document: {scores}")
        print(f"Document ID: {r.get('doc_id', 'N/A')}")
    else:
        print("Cannot calculate Rouge score for 10th document: Empty or NaN answers")
except Exception as e:
    print(f"Error calculating Rouge score for 10th document: {e}")

# Calculate Rouge scores for all pairs
rouge1_f1_scores = []
skipped_count = 0

for idx, row in df_results.iterrows():
    try:
        # Handle potential NaN values
        llm_answer = row.answer_llm if pd.notna(row.answer_llm) else ''
        orig_answer = row.answer_orig if pd.notna(row.answer_orig) else ''
        
        # Skip empty answers
        if not llm_answer or not orig_answer:
            skipped_count += 1
            continue
            
        scores = rouge_scorer.get_scores(llm_answer, orig_answer)[0]
        rouge1_f1_scores.append(scores['rouge-1']['f'])
    except Exception as e:
        skipped_count += 1
        # Print error for debugging but only for the first few occurrences
        if skipped_count <= 5:
            print(f"Error calculating Rouge score for row {idx}: {e}")

# Calculate average Rouge-1 F1 score
if rouge1_f1_scores:
    avg_rouge1_f1 = np.mean(rouge1_f1_scores)
    print(f"Average Rouge-1 F1 score: {avg_rouge1_f1}")
    print(f"Calculated scores for {len(rouge1_f1_scores)} out of {len(df_results)} rows")
    print(f"Skipped {skipped_count} rows due to errors or empty answers")
else:
    print("No valid Rouge scores calculated")
```

Rouge score `0.35` is closest to our result `0.3517`.


