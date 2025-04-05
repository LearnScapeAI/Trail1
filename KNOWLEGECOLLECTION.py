import requests
import json
import time
from bs4 import BeautifulSoup
import uuid
from pinecone import Pinecone, ServerlessSpec, PineconeApiException

# -------------------------
# Configuration Parameters
# -------------------------
# Google Custom Search configuration
GOOGLE_API_KEY = 'AIzaSyBiKOLhuLP8ZLZeSsNx0ufghUDrTucTzug'
CUSTOM_SEARCH_ENGINE_ID = '627f547ffe4c94a8d'
GEMINI_API_KEY = 'YAIzaSyA8zJq4rCy_fY28QRBchRY6a3paSASTZeE'  # Replace with your actual Gemini API key

# Pinecone configuration
PINECONE_API_KEY = "pcsk_3JheQq_CDgQXKpDcTp1FeiTxascJDRJqmTp4YtDqFWATY4vJFnsLNJVKYjqM4QesD8agJW"
INDEX_NAME = "knowledge-base-index"
EMBEDDING_DIM = 512  # Adjust based on your embedding model

# Initialize Pinecone using the serverless client
pc = Pinecone(api_key=PINECONE_API_KEY)

# -------------------------
# Google Custom Search Functions
# -------------------------
def google_search(query, api_key, cse_id, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'num': num_results
    }
    response = requests.get(search_url, params=params)
    results = response.json()
    return results.get("items", [])

# -------------------------
# Web Scraping Helpers
# -------------------------
def fetch_page_content(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {url} (status code {response.status_code})")
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator=" ")

# -------------------------
# Gemini API Call (Placeholder)
# -------------------------
def call_gemini_api(prompt, content):
    """
    Placeholder for the Gemini API call.
    Replace this with an actual request to Gemini that returns structured data.
    """
    return {
        "title": "Extracted Title",
        "summary": "This is a summary extracted from the content.",
        "key_points": ["Point 1", "Point 2", "Point 3"]
    }

# -------------------------
# Building the Knowledge Base
# -------------------------
def build_knowledge_base(query):
    search_results = google_search(query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID)
    knowledge_base = []
    for item in search_results:
        link = item.get("link")
        print(f"Processing: {link}")
        html_content = fetch_page_content(link)
        if html_content:
            text_content = extract_text_from_html(html_content)
            prompt = (
                f"Extract the main title, a summary, and key points from the text "
                f"about {query}. Provide the results in JSON format with keys 'title', "
                f"'summary', and 'key_points'."
            )
            structured_data = call_gemini_api(prompt, text_content)
            structured_data["source_url"] = link
            knowledge_base.append(structured_data)
    return knowledge_base

# -------------------------
# Embedding Generation (Updated Placeholder)
# -------------------------
def generate_embedding(text):
    """
    Dummy embedding generation function.
    For testing, this returns a vector filled with 0.1's instead of zeros.
    Replace with your actual embedding generation code.
    """
    return [0.1] * EMBEDDING_DIM  # Now returns a non-zero vector

# -------------------------
# Pinecone Integration
# -------------------------
def upsert_to_pinecone(knowledge_base):
    if not knowledge_base:
        print("Knowledge base is empty. Skipping upsert.")
        return

    try:
        if INDEX_NAME not in pc.list_indexes():
            print("Index does not exist. Creating index...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{INDEX_NAME}' created. Waiting for index to be ready...")
            while True:
                index_description = pc.describe_index(INDEX_NAME)
                if index_description["status"]["ready"]:
                    break
                time.sleep(5)
        else:
            print(f"Index '{INDEX_NAME}' already exists.")
    except PineconeApiException as e:
        if "ALREADY_EXISTS" in str(e):
            print(f"Index '{INDEX_NAME}' already exists.")
        else:
            raise e

    index = pc.Index(INDEX_NAME)
    vectors = []
    for item in knowledge_base:
        vector_id = str(uuid.uuid4())
        embedding = generate_embedding(item["summary"])
        metadata = {
            "title": item["title"],
            "summary": item["summary"],
            "source_url": item["source_url"],
            "key_points": item["key_points"]
        }
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    try:
        index.upsert(vectors=vectors, namespace="default")
        print("Upserted data to Pinecone.")
    except Exception as e:
        print(f"Error upserting data to Pinecone: {e}")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    topic = input("Enter the topic for knowledge base creation: ")
    kb = build_knowledge_base(topic)
    print("Knowledge Base Created:")
    print(json.dumps(kb, indent=4))
    upsert_to_pinecone(kb)
