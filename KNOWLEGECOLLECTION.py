import requests
import json
import time
from bs4 import BeautifulSoup
import pinecone
import uuid  # For generating unique IDs for each record

# -------------------------
# Configuration Parameters
# -------------------------
# Google Custom Search configuration
GOOGLE_API_KEY = 'ca1b37751147709c8ea4fa1e5bedd95fc67aff80'
CUSTOM_SEARCH_ENGINE_ID = '627f547ffe4c94a8d'
GEMINI_API_KEY = 'YAIzaSyA8zJq4rCy_fY28QRBchRY6a3paSASTZeE'  # Replace with your actual Gemini API key

# Pinecone configuration
pc = pinecone(api_key ="ppcsk_74kXqY_DxuwPhoDP8jtwdwc8944wr3iLeMvSamyZpbpbJjvLC5u5KRxx5WPP5vafWD5s7v",)
#pinecone.init(api_key='ppcsk_74kXqY_DxuwPhoDP8jtwdwc8944wr3iLeMvSamyZpbpbJjvLC5u5KRxx5WPP5vafWD5s7v', environment='us-east1-gcp')
'''PINECONE_API_KEY = 'ppcsk_74kXqY_DxuwPhoDP8jtwdwc8944wr3iLeMvSamyZpbpbJjvLC5u5KRxx5WPP5vafWD5s7v'
PINECONE_ENVIRONMENT = 'us-east1-gcp'  # e.g., 'us-east1-gcp'
PINECONE_INDEX_NAME = 'knowledge-base-index'
EMBEDDING_DIM = 512  # Adjust this to match the dimensionality of your embeddings'''

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
    items = results.get("items", [])
    return items

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
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator=" ")
    return text

# -------------------------
# Gemini API Placeholder
# -------------------------
def call_gemini_api(prompt, content):
    """
    Placeholder function to simulate a call to the Gemini API.
    Replace this with the actual API call as per Gemini's documentation.
    """
    # For demonstration, we return dummy structured data:
    return {
        "title": "Extracted Title",
        "summary": "This is a summary extracted from the content.",
        "key_points": ["Point 1", "Point 2", "Point 3"]
    }

# -------------------------
# Building the Knowledge Base
# -------------------------
def build_knowledge_base(query):
    # Step 1: Get search results from Google Custom Search API
    search_results = google_search(query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID)
    knowledge_base = []
    
    # Step 2: Process each search result
    for item in search_results:
        link = item.get("link")
        print(f"Processing: {link}")
        html_content = fetch_page_content(link)
        if html_content:
            text_content = extract_text_from_html(html_content)
            # Step 3: Use Gemini API to extract structured information
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
# Embedding Generation (Placeholder)
# -------------------------
def generate_embedding(text):
    """
    Dummy embedding generation function.
    Replace with your actual embedding model to generate a vector representation of 'text'.
    """
    return [0.0] * EMBEDDING_DIM  # Replace with real embeddings

# -------------------------
# Pinecone Integration
# -------------------------
def upsert_to_pinecone(knowledge_base):
    # Initialize Pinecone
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    except AttributeError as e:
        print("Error: Pinecone initialization failed. Ensure you have installed the correct Pinecone client (pip install pinecone-client) and that there is no naming conflict (e.g., a local file named pinecone.py).")
        raise e
    
    # Create the index if it doesn't exist
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        print("Index does not exist. Creating index...")
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM)
        # Wait until the index is ready
        print("Waiting for index to be ready...")
        while True:
            index_description = pinecone.describe_index(PINECONE_INDEX_NAME)
            if index_description["status"]["ready"]:
                break
            time.sleep(5)
    
    # Connect to the index
    index = pinecone.Index(PINECONE_INDEX_NAME)
    
    # Prepare data for upsert
    vectors = []
    for item in knowledge_base:
        # Generate a unique ID for the item
        vector_id = str(uuid.uuid4())
        # Generate an embedding from the summary (or choose another text field)
        embedding = generate_embedding(item["summary"])
        # Metadata to store along with the vector
        metadata = {
            "title": item["title"],
            "summary": item["summary"],
            "source_url": item["source_url"],
            "key_points": item["key_points"]
        }
        vectors.append((vector_id, embedding, metadata))
    
    try:
        index.upsert(vectors=vectors)
        print("Upserted data to Pinecone.")
    except Exception as e:
        print(f"Error upserting data to Pinecone: {e}")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Prompt the user for a topic
    topic = input("Enter the topic for knowledge base creation: ")
    
    # Build the knowledge base from web content
    kb = build_knowledge_base(topic)
    
    # Save the knowledge base to a JSON file
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=4)
    
    print("Knowledge Base Created:")
    print(json.dumps(kb, indent=4))
    
    # Upsert the knowledge base into Pinecone
    upsert_to_pinecone(kb)
