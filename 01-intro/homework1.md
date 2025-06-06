**Q1. Running Elastic**
First, we need to run Elasticsearch 8.17.6 using Podman (I use Podman more frequently lately instead of Docker)
Open VS Code's terminal

```bash
podman run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.17.6
```
Once Elasticsearch is running, we can get the cluster information:
```bash
curl localhost:9200
```
The response will contain the version.build_hash value. 

```
"build_hash" : "dbcbbbd0bc4924cfeb28929dc05d82d662c527b7"
```

**Q2. Indexing the data**

Install the ElasticSearch client for Python:
```python
pip install elasticsearch
```
Install requests library:
```python
pip install requests
```
Open notebook and run Python script:
```python
import requests 
from elasticsearch import Elasticsearch
import json

# Get the documents
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# Display the structure of the data (helpful in VS Code)
print(f"Number of courses: {len(documents_raw)}")
print(f"Sample course: {documents_raw[0]['course']}")
print(f"Sample document: {json.dumps(documents_raw[0]['documents'][0], indent=2)}")

# Process the documents
documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(f"Total documents: {len(documents)}")
```
Now, let's create the index with proper mappings and index the data:

```python
# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Check if connection is successful
if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Could not connect to Elasticsearch")
    # If you're having connection issues, make sure Elasticsearch is running
    # and check if there are any network restrictions

# Define index settings
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

# Create the index
index_name = "faq_documents"

# Delete the index if it already exists (optional)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Deleted existing index: {index_name}")

# Create the index
es.indices.create(index=index_name, body=index_settings)
print(f"Created index: {index_name}")

# Index the documents
for i, doc in enumerate(documents):
    es.index(index=index_name, id=i, document=doc)
    
    # Print progress every 50 documents (helpful in VS Code)
    if (i + 1) % 50 == 0:
        print(f"Indexed {i + 1}/{len(documents)} documents")

print(f"Indexed all {len(documents)} documents")
```

The function used to add data to Elasticsearch is `index`.

**Q3. Searching**
Let's search for "How do execute a command on a Kubernetes pod?":
```python
query = "How do execute a command on a Kubernetes pod?"
search_query = {
    "size": 5,
    "query": {
        "multi_match": {
            "query": query,
            "fields": ["question^4", "text"],
            "type": "best_fields"
        }
    }
}
response = es.search(index=index_name, body=search_query)
# Print the results in a readable format (helpful in VS Code)
print(f"Total hits: {response['hits']['total']['value']}")
print("\nTop 5 results:")
for i, hit in enumerate(response['hits']['hits']):
    print(f"\nResult {i+1}:")
    print(f"Score: {hit['_score']}")
    print(f"Course: {hit['_source']['course']}")
    print(f"Question: {hit['_source']['question']}")
    print(f"Text: {hit['_source']['text'][:100]}...")  # Show first 100 chars
# Print the top result score for Q3
top_score = response['hits']['hits'][0]['_score']
print(f"\nTop score (answer to Q3): {top_score}")
```
The score for the top ranking result will be displayed `44.50556`. It is close to `44.50`

**Q4. Filtering**
Now let's search for "How do copy a file to a Docker container?" but only in machine-learning-zoomcamp:

```python
query = "How do copy a file to a Docker container?"
search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            }
        }
    }
}
response = es.search(index=index_name, body=search_query)
# Print the results in a readable format
print(f"Total hits: {response['hits']['total']['value']}")
print("\nTop 3 results from machine-learning-zoomcamp:")
for i, hit in enumerate(response['hits']['hits']):
    print(f"\nResult {i+1}:")
    print(f"Score: {hit['_score']}")
    print(f"Question: {hit['_source']['question']}")
    print(f"Text: {hit['_source']['text'][:100]}...")  # Show first 100 chars
# Store the results for use in Q5
filtered_results = response['hits']['hits']
# Print the 3rd question specifically for Q4
if len(filtered_results) >= 3:
    third_question = filtered_results[2]['_source']['question']
    print(f"\nThird question (answer to Q4): {third_question}")
else:
    print("\nLess than 3 results returned")
```
Third question is `How do I copy files from a different folder into docker container’s working directory?`

**Q5. Building a prompt**
Let's build the prompt using the results from Q4:
```python
# Build context using the template
context_template = """
Q: {question}
A: {text}
""".strip()
context_entries = []
for hit in filtered_results:
    context_entry = context_template.format(
        question=hit['_source']['question'],
        text=hit['_source']['text']
    )
    context_entries.append(context_entry)
# Join context entries with double linebreaks
context = "\n\n".join(context_entries)
# Build the final prompt
question = "How do I execute a command in a running docker container?"
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
QUESTION: {question}
CONTEXT:
{context}
""".strip()
prompt = prompt_template.format(question=question, context=context)
# Print the prompt (helpful to see what we've created)
print("Generated prompt:")
print("=" * 80)
print(prompt)
print("=" * 80)
# Calculate the length of the prompt
prompt_length = len(prompt)
print(f"\nPrompt length (answer to Q5): {prompt_length} characters")
```
The length of the resulting prompt is `1460` which is close to `1446` from options.

**Q6. Tokens**
Let's calculate the number of tokens in our prompt:

```python
# Install tiktoken if not already installed
# !pip install tiktoken
import tiktoken
# Get the encoding for GPT-4o
encoding = tiktoken.encoding_for_model("gpt-4o")
# Encode the prompt to get tokens
tokens = encoding.encode(prompt)
# Count the number of tokens
token_count = len(tokens)
print(f"Number of tokens (answer to Q6): {token_count}")
# Optional: Display some token examples
print("\nSample tokens and their decoded values:")
for i in range(min(5, len(tokens))):
    token = tokens[i]
    decoded = encoding.decode_single_token_bytes(token)
    print(f"Token {token}: {decoded}")
```

Prompt have `322` tokens which is close to option `320`