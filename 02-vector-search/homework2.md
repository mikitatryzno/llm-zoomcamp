First, we need to install the required packages. Run these commands in your terminal (not in a Python cell):

```bash
# Install required Python packages
pip install -q "qdrant-client[fastembed]>=1.14.2" fastembed
```
Then start Qdrant using Podman instead of Docker:

```bash
# Pull and run Qdrant using Podman instead of Docker
podman pull qdrant/qdrant
podman run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```
Now let's solve each question in Python cells:

**Q1. Embedding the query**

```python
from fastembed import TextEmbedding
# Initialize the embedding model
embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-small-en")
# Embed the query
query = "I just discovered the course. Can I join now?"
query_embedding = list(embedding_model.embed([query]))[0]
# Check the shape of the embedding
print(f"Shape of embedding: {query_embedding.shape}")  # Should be (512,)
# Find the minimal value in the array
min_value = query_embedding.min()
print(f"Minimum value in the embedding: {min_value}")
```
After running this code, minimum value in the embedding: `-0.11726373885183883`

**Q2. Cosine similarity with another vector**

```python
# Embed the document
doc = 'Can I still join the course after the start date?'
doc_embedding = list(embedding_model.embed([doc]))[0]
# Compute cosine similarity (dot product of normalized vectors)
import numpy as np
cosine_similarity = query_embedding.dot(doc_embedding)
# Verify that vectors are normalized
print(f"Query vector norm: {np.linalg.norm(query_embedding)}")  # Should be close to 1.0
print(f"Document vector norm: {np.linalg.norm(doc_embedding)}")  # Should be close to 1.0
print(f"Cosine similarity: {cosine_similarity}")
```
Cosine similarity: `0.9008528895674548`

**Q3. Ranking by cosine**

```python
documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  "Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon't forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]
# Embed all document texts
doc_texts = [doc['text'] for doc in documents]
doc_embeddings = list(embedding_model.embed(doc_texts))
# Create a matrix of document embeddings
import numpy as np
V = np.vstack(doc_embeddings)
# Compute cosine similarities with the query
similarities = V.dot(query_embedding)
# Find the document with highest similarity
highest_idx = np.argmax(similarities)
highest_similarity = similarities[highest_idx]
print(f"Document with highest similarity: {highest_idx}")
print(f"Similarity score: {highest_similarity}")
print(f"Document text: {documents[highest_idx]['text'][:100]}...")
```
Document with highest similarity is `1`
Similarity score is `0.8182378150042889`
Document text is `Yes, we will keep all the materials after the course finishes, so you can follow the course at your .`

**Q4. Ranking by cosine, version two**

```python
# Create full_text by concatenating question and text
full_texts = [doc['question'] + ' ' + doc['text'] for doc in documents]
# Embed the full texts
full_text_embeddings = list(embedding_model.embed(full_texts))
# Create a matrix of full text embeddings
V_full = np.vstack(full_text_embeddings)
# Compute cosine similarities with the query
similarities_full = V_full.dot(query_embedding)
# Find the document with highest similarity
highest_idx_full = np.argmax(similarities_full)
highest_similarity_full = similarities_full[highest_idx_full]
print(f"Document with highest similarity (full text): {highest_idx_full}")
print(f"Similarity score: {highest_similarity_full}")
print(f"Document question: {documents[highest_idx_full]['question']}")
```
The answer is the index of the document with the highest similarity score when using the full text.
Document with highest similarity (full text): `0`, 
Similarity score: `0.8514543236908068`
This demonstrates why RAG systems often benefit from carefully considering which fields to include in the embedding process, as different combinations can yield different retrieval results.

**Q5. Selecting the embedding model**
```python
from fastembed import TextEmbedding
# List available models
available_models = TextEmbedding.list_supported_models()
print("Available models:")
for model in available_models:
    print(model)
# Check dimensions of a smaller model
small_model = TextEmbedding(model_name="BAAI/bge-small-en")
test_embedding = list(small_model.embed(["test"]))[0]
print(f"Dimension of BAAI/bge-small-en: {test_embedding.shape[0]}")
```

Based on the output, the smallest dimensionality available in fastembed models is `384`

**Q6. Indexing with qdrant**
```python
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# Download documents
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
# Filter for ML Zoomcamp documents
documents = []
for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(f"Number of ML Zoomcamp documents: {len(documents)}")
# Initialize the small embedding model
small_model = TextEmbedding(model_name="BAAI/bge-small-en")
# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
# Create a collection
collection_name = "ml_zoomcamp_faq"
vector_size = list(small_model.embed(["test"]))[0].shape[0]  # Should be 384
# Create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"Collection {collection_name} already exists")
except:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection {collection_name}")
# Prepare documents for indexing
points = []
for i, doc in enumerate(documents):
    # Combine question and text
    text = doc['question'] + ' ' + doc['text']
    
    # Embed the text
    embedding = list(small_model.embed([text]))[0]
    
    # Create a point
    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload=doc
    )
    points.append(point)

# Upload points in batches
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(
        collection_name=collection_name,
        points=batch
    )
    print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
# Query the collection
query = "I just discovered the course. Can I join now?"
query_embedding = list(small_model.embed([query]))[0]
search_results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=5
)
# Print the top result and its score
print(f"Top result score: {search_results[0].score}")
print(f"Question: {search_results[0].payload['question']}")
The answer is the highest score value from the search results.
```

Top result score is `0.8703172`.