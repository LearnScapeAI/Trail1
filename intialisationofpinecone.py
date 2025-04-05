from pinecone import Pinecone, ServerlessSpec

# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_74kXqQ_DxuwPhoDP8jtwdwc8944wr3iLeMvSamyZpbpbJjvLC5u5KRxx5WPP5vafWD5s7vy")

# Create a serverless index named "quickstart"
index_name = "quickstart"
pc.create_index(
    name=index_name,
    dimension=2,  # Replace with your model dimensions
    metric="cosine",  # Replace with your chosen metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print(f"Index '{index_name}' created.")

# Upsert sample data
vector_id = "example-id"
vector = [0.1, 0.2]  # Replace with your actual 2-dimensional embedding
metadata = {
    "title": "Example Title",
    "summary": "Example summary",
    "source_url": "https://example.com"
}
pc.index(index_name).upsert(vectors=[(vector_id, vector, metadata)])
print("Data upserted to Pinecone.")

# Query the index (optional)
query_response = pc.index(index_name).query(queries=[[0.1, 0.2]], top_k=1, include_metadata=True)
print("Query response:", query_response)
