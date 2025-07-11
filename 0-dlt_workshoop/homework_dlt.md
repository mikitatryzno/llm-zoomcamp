**Setup Environment**

First, let's install the required packages:
```bash
pip install -q "dlt[qdrant]" "qdrant-client[fastembed]"
```

**Question 1. dlt Version**
To find out the installed version of dlt, run this Python code:

```python
import dlt
print(f"dlt version: {dlt.__version__}")
```

The version of dlt that were installed is `1.13`

**Question 2. dlt pipeline**

Let's create a complete script to load the FAQ data into Qdrant using dlt:

```python
import dlt
import requests
from dlt.destinations import qdrant

# Step 1: Create DLT resource
@dlt.resource(write_disposition="replace", name="zoomcamp_data")
def zoomcamp_data():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            yield doc

# Step 2: Create Qdrant destination
qdrant_destination = qdrant(
  qd_path="db.qdrant", 
)

# Step 3: Create and run the pipeline
pipeline = dlt.pipeline(
    pipeline_name="zoomcamp_pipeline",
    destination=qdrant_destination,
    dataset_name="zoomcamp_tagged_data"
)

# Step 4: Run the pipeline
load_info = pipeline.run(zoomcamp_data())
print(pipeline.last_trace)
```

After running this code, we should look at the output for a line that says "Normalized data for the following tables:" followed by information about the zoomcamp_data collection. The number of rows inserted is `948`

**Question 3. Embeddings**

To find out which embedding model was used, we need to examine the meta.json file in the db.qdrant folder:

```python
import json
import os

# Path to the meta.json file
meta_file_path = os.path.join("db.qdrant", "meta.json")

# Read and display the contents
with open(meta_file_path, 'r') as f:
    meta_data = json.load(f)
    print(json.dumps(meta_data, indent=2))
```

The name of the embedding model is `fast-bge-small-en`