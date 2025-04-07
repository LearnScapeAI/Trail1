from dotenv import load_dotenv
import os
import requests
import json
import time
import uuid
import sys
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec, PineconeApiException

load_dotenv()  # Load environment variables from the .env file

# -------------------------
# Configuration Parameters
# -------------------------

# Load API Keys from Environment Variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', "pcsk_3JheQq_CDgQXKpDcTp1FeiTxascJDRJqmTp4YtDqFWATY4vJFnsLNJVKYjqM4QesD8agJW")
CUSTOM_SEARCH_ENGINE_ID = os.getenv('CUSTOM_SEARCH_ENGINE_ID', '627f547ffe4c94a8d')

# Validate API Keys
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set. This is needed for Google Custom Search.")
    sys.exit(1)
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set. This is needed for the Gemini API.")
    sys.exit(1)
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY environment variable not set or provided.")
    sys.exit(1)
if not CUSTOM_SEARCH_ENGINE_ID:
    print("Error: CUSTOM_SEARCH_ENGINE_ID environment variable not set or provided.")
    sys.exit(1)


# Pinecone configuration
INDEX_NAME = "knowledge-base-index"
EMBEDDING_DIM = 384  # Adjust based on your embedding model

# Initialize Pinecone using the serverless client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Gemini Configuration
MODEL_NAME = "gemini-1.5-pro-latest"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"


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
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        results = response.json()
        return results.get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"Error during Google Search API call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            print(f"Response Text: {e.response.text}")
        return []  # Return empty list on error


# -------------------------
# Web Scraping Helpers
# -------------------------
def fetch_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)  # Increased timeout slightly
        if response.status_code == 200:
            # Attempt to detect encoding, fallback to utf-8
            response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'
            return response.text
        else:
            print(f"Failed to fetch {url} (status code {response.status_code})")
            return ""
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching {url}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {url}: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
        return ""


def extract_text_from_html(html_content):
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):  # Added common non-content tags
            element.extract()
        # Get text, replace multiple whitespace chars with a single space
        text = soup.get_text(separator=" ", strip=True)
        return ' '.join(text.split())
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""


# -------------------------
# Gemini API Call
# -------------------------
def call_gemini_api(prompt, document_content):
    """
    Calls the Gemini API using the correct endpoint and payload structure,
    then returns a dictionary with title, summary, and key_points.

    Args:
        prompt (str): The instruction or question for the model.
        document_content (str): The document or text to process.

    Returns:
        dict: Contains 'title', 'summary', and 'key_points' from the API response,
              or error information if something went wrong.
    """
    headers = {
        "Content-Type": "application/json"
    }

    # Limit document content size to avoid exceeding API limits (adjust as needed)
    max_doc_length = 30000  # Example limit (characters)
    if len(document_content) > max_doc_length:
        print(f"Warning: Document content truncated to {max_doc_length} characters for Gemini API call.")
        document_content = document_content[:max_doc_length]

    combined_text = f"Prompt: {prompt}\n\nDocument:\n{document_content}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": combined_text}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 8192
        }
    }

    print("--- Calling Gemini API ---")
    print(f"URL: {GEMINI_API_URL.split('?')[0]}?key=...")  # Hide key in log

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)
        result = response.json()

        # Robust extraction of generated text
        try:
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as parse_err:
            error_info = f"Could not extract generated text from API response structure. Error: {parse_err}."
            prompt_feedback = result.get("promptFeedback")
            if prompt_feedback:
                block_reason = prompt_feedback.get("blockReason")
                safety_ratings = prompt_feedback.get("safetyRatings")
                error_info += f" Prompt Feedback: BlockReason={block_reason}, SafetyRatings={safety_ratings}"
            else:
                candidates = result.get("candidates", [])
                if not candidates:
                    error_info += " The 'candidates' list in the response was empty or missing."
                elif candidates[0].get("finishReason") not in [None, "STOP", "MAX_TOKENS"]:
                    error_info += f" Candidate finishReason was '{candidates[0].get('finishReason')}'."
            print(f"Warning: {error_info}")
            print("Full Response:", json.dumps(result, indent=2))
            return {
                "title": "Error",
                "summary": error_info,
                "key_points": []
            }

        # Attempt to parse the generated text as JSON
        try:
            # Handle potential markdown code block ```json ... ```
            if generated_text.strip().startswith("```json"):
                generated_text = generated_text.strip()[7:-3].strip()
            elif generated_text.strip().startswith("```"):
                generated_text = generated_text.strip()[3:-3].strip()

            parsed_output = json.loads(generated_text)
            title = parsed_output.get("title", "No title extracted")
            summary = parsed_output.get("summary", "No summary extracted")
            key_points = parsed_output.get("key_points", [])
            # Ensure key_points is a list
            if not isinstance(key_points, list):
                print(f"Warning: 'key_points' from Gemini was not a list, using empty list. Value: {key_points}")
                key_points = []
            return {"title": title, "summary": summary, "key_points": key_points}
        except json.JSONDecodeError:
            print("Warning: Gemini response was not valid JSON. Using the full text as summary.")
            # Return the raw text if it's not JSON, assuming it might be the summary itself
            return {"title": "Title Generation Failed (Non-JSON)", "summary": generated_text, "key_points": []}

    except requests.exceptions.Timeout as e:
        print(f"Error calling Gemini API (Timeout): {e}")
        return {
            "title": "Error",
            "summary": f"API request timed out after 120 seconds: {e}",
            "key_points": []
        }
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API (RequestException): {e}")
        error_content = "No response content available"
        status_code = "N/A"
        if hasattr(e, 'response') and e.response is not None:
            error_content = e.response.text
            status_code = e.response.status_code
        print(f"Response status: {status_code}")
        print(f"Response content: {error_content}")
        return {
            "title": "Error",
            "summary": f"API request failed: {e}. Status: {status_code}",
            "key_points": []
        }
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e.__class__.__name__} - {e}")
        error_summary = f"An unexpected error occurred: {e}"
        try:
            if 'response' in locals() and response is not None:
                error_summary += f" | Status: {response.status_code} | Response Text: {response.text[:500]}..."
        except Exception:
            pass
        return {
            "title": "Error",
            "summary": error_summary,
            "key_points": []
        }


# -------------------------
# Building the Knowledge Base
# -------------------------
def build_knowledge_base(query):
    # Use the globally loaded keys and IDs
    search_results = google_search(query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID)
    knowledge_base = []
    if not search_results:
        print("No search results found for the query.")
        return []

    for item in search_results:
        link = item.get("link")
        title_from_search = item.get("title", "N/A")  # Get title from search result as fallback
        if not link:
            print("Search result item missing 'link'. Skipping.")
            continue

        print(f"\nProcessing: {link} ({title_from_search})")
        html_content = fetch_page_content(link)
        if not html_content:
            print(f"Skipping {link} due to fetch error.")
            continue

        text_content = extract_text_from_html(html_content)
        if not text_content or len(text_content) < 100:  # Skip if very little text extracted
            print(f"Skipping {link} due to insufficient text content ({len(text_content)} chars).")
            continue

        print(f"Extracted ~{len(text_content)} characters of text. Calling Gemini...")

        # Updated prompt for clarity and robustness
        prompt = f"""Analyze the following document content which is related to the topic '{query}'.
Provide the output STRICTLY in JSON format with the following keys:
- "title": A concise and accurate title for the document's content.
- "summary": A comprehensive summary of the main information (around 150-250 words).
- "key_points": A list of 3-7 bullet points highlighting the most important facts or conclusions.

If the document content is irrelevant to '{query}', very short, or seems like an error page, return JSON with "title": "Extraction Failed", "summary": "Content irrelevant or insufficient", and "key_points": [].

JSON Output:
"""
        structured_data = call_gemini_api(prompt, text_content)

        # Basic validation of the result from Gemini
        if structured_data.get("title") == "Error" or structured_data.get("title") == "Extraction Failed":
            print(f"Gemini extraction failed or content was irrelevant for {link}. Summary: {structured_data.get('summary')}")
            # Optionally skip adding failed extractions to the KB
            # continue
        else:
            print(f"Successfully processed: Title - {structured_data.get('title')}")

        # Add source URL regardless of success/failure to track origin
        structured_data["source_url"] = link
        knowledge_base.append(structured_data)

        # Optional: Add a delay between processing URLs
        time.sleep(0.5)

    return knowledge_base


# -------------------------
# Embedding Generation (Updated Placeholder)
# -------------------------
def generate_embedding(text):
    """
    Dummy embedding generation function.
    For testing, this returns a vector filled with 0.1's instead of zeros.
    Replace with your actual embedding generation code (e.g., using Sentence Transformers, OpenAI API, etc.).
    """
    if not text or not isinstance(text, str):  # Basic check
        print("Warning: generate_embedding received invalid text. Returning zero vector.")
        return [0.0] * EMBEDDING_DIM
    # Return a simple non-zero vector for testing Pinecone upsert
    return [0.1 + (hash(text) % 10) * 0.01] * EMBEDDING_DIM  # Slightly varied dummy vector


# -------------------------
# Pinecone Integration
# -------------------------
def upsert_to_pinecone(knowledge_base):
    if not knowledge_base:
        print("Knowledge base is empty. Skipping Pinecone upsert.")
        return

    valid_items_for_upsert = [
        item for item in knowledge_base
        if item.get("title") != "Error" and item.get("summary") and item.get("source_url")
    ]

    if not valid_items_for_upsert:
        print("No valid items extracted for Pinecone upsert.")
        return

    print(f"\nAttempting to upsert {len(valid_items_for_upsert)} items to Pinecone index '{INDEX_NAME}'...")

    try:
        # Check if the index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"Index '{INDEX_NAME}' does not exist. Creating index...")
            try:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Ensure region is correct
                )
                print(f"Index '{INDEX_NAME}' creation initiated. Waiting for it to be ready...")
                # Wait for index to be ready
                while True:
                    try:
                        index_description = pc.describe_index(INDEX_NAME)
                        if index_description and index_description["status"]["ready"]:
                            print(f"Index '{INDEX_NAME}' is ready.")
                            break
                    except PineconeApiException as desc_e:
                        print(f"Waiting for index... (describe error: {desc_e})")
                    time.sleep(10)  # Wait longer between checks
            except PineconeApiException as create_e:
                print(f"Error creating Pinecone index: {create_e}")
                if "quota" in str(create_e).lower():
                    print("Suggestion: Check your Pinecone plan limits and existing indexes.")
                return  # Stop if index creation fails
        else:
            print(f"Index '{INDEX_NAME}' already exists.")

        index = pc.Index(INDEX_NAME)
        print("Preparing vectors for upsert...")
        vectors_to_upsert = []
        ids_generated = set()

        for item in valid_items_for_upsert:
            text_to_embed = item.get("summary", "")
            if not text_to_embed:
                print(f"Warning: Skipping item with empty summary. URL: {item.get('source_url')}")
                continue

            embedding = generate_embedding(text_to_embed)
            vector_id = f"urlhash-{hash(item.get('source_url'))}"
            if vector_id in ids_generated:
                print(f"Warning: Duplicate ID generated ({vector_id}), potentially from the same URL. Appending UUID.")
                vector_id = f"{vector_id}-{uuid.uuid4()}"
            ids_generated.add(vector_id)

            metadata = {
                "title": str(item.get("title", "N/A"))[:500],
                "summary": str(text_to_embed)[:2000],
                "source_url": str(item.get("source_url", "N/A")),
                "key_points": json.dumps(item.get("key_points", []))
            }

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        if not vectors_to_upsert:
            print("No valid vectors generated for upsert.")
            return

        batch_size = 100
        print(f"Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                print(f"Upserting batch {i // batch_size + 1}...")
                index.upsert(vectors=batch, namespace="default")
                print(f"Batch {i // batch_size + 1} upserted successfully.")
            except PineconeApiException as upsert_e:
                print(f"Error upserting batch {i // batch_size + 1} to Pinecone: {upsert_e}")
                print(f"Problematic batch (first item ID): {batch[0]['id'] if batch else 'N/A'}")
                break
            except Exception as e:
                print(f"Unexpected error during batch upsert: {e}")
                break
        print("Pinecone upsert process completed.")

    except PineconeApiException as e:
        print(f"A Pinecone API error occurred: {e}")
        if "not found" in str(e).lower() and "index" in str(e).lower():
            print(f"Error: Index '{INDEX_NAME}' not found or access denied. Please check the index name and API key permissions.")
        elif "authentication" in str(e).lower():
            print("Error: Pinecone authentication failed. Check your PINECONE_API_KEY.")
        else:
            if hasattr(e, 'body'):
                print(f"Error details: {e.body}")

    except Exception as e:
        print(f"An unexpected error occurred during Pinecone operations: {e}")


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    topic = input("Enter the topic for knowledge base creation: ")
    if not topic:
        print("Topic cannot be empty.")
    else:
        print(f"\n--- Starting Knowledge Base Creation for: {topic} ---")
        kb = build_knowledge_base(topic)

        print("\n--- Knowledge Base Creation Complete ---")
        print(f"Generated {len(kb)} entries for the knowledge base.")

        if kb:
            upsert_to_pinecone(kb)
        else:
            print("No entries generated, skipping Pinecone upsert.")

        print("\n--- Script Finished ---")
