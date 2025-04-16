from dotenv import load_dotenv
import os
import requests
import json
import time
import uuid
import sys
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from datetime import datetime, timedelta

# Load environment variables from the .env file
load_dotenv()

# -------------------------
# Configuration Parameters
# -------------------------

# Load API Keys from Environment Variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', "your_default_pinecone_key")
CUSTOM_SEARCH_ENGINE_ID = os.getenv('CUSTOM_SEARCH_ENGINE_ID', '627f547ffe4c94a8d')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')  # New: YouTube API Key

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
if not YOUTUBE_API_KEY:
    print("Warning: YOUTUBE_API_KEY environment variable not set. YouTube integration will be skipped.")

# Pinecone configuration
INDEX_NAME = "knowledge-base-index"
EMBEDDING_DIM = 384  # Ensure this matches your Sentence Transformer model

# Initialize Pinecone using the serverless client
pc = Pinecone(api_key=PINECONE_API_KEY)

# -------------------------
# Sentence Transformer Initialization
# -------------------------
from sentence_transformers import SentenceTransformer

# Instantiate the embedding model once at startup.
# "all-MiniLM-L6-v2" outputs 384-dimensional embeddings.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Gemini Configuration
# -------------------------
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
        return []

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
        response = requests.get(url, headers=headers, timeout=15)
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
        # Remove script, style, nav, footer, and aside elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.extract()
        text = soup.get_text(separator=" ", strip=True)
        return ' '.join(text.split())
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""

# -------------------------
# New: YouTube Search Helper Function
# -------------------------
def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    """
    Given a subtopic query, search YouTube using the Data API and return the first video
    that meets the criteria (duration < 600 seconds). Returns a dictionary with relevant video info.
    """
    import isodate  # To parse ISO 8601 duration strings

    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "snippet",
        "q": subtopic_query,
        "key": api_key,
        "maxResults": max_results,
        "type": "video"
    }
    try:
        search_response = requests.get(search_url, params=params)
        search_response.raise_for_status()
        results = search_response.json().get("items", [])
    except Exception as e:
        print(f"Error during YouTube search API call: {e}")
        return None

    for item in results:
        video_id = item["id"]["videoId"]
        params_details = {
            "part": "contentDetails,snippet",
            "id": video_id,
            "key": api_key
        }
        try:
            details_response = requests.get(video_url, params=params_details)
            details_response.raise_for_status()
            details_items = details_response.json().get("items", [])
        except Exception as e:
            print(f"Error retrieving YouTube video details: {e}")
            continue

        if details_items:
            content_details = details_items[0].get("contentDetails", {})
            duration_str = content_details.get("duration")
            try:
                duration_seconds = isodate.parse_duration(duration_str).total_seconds()
            except Exception as e:
                print(f"Error parsing duration: {e}")
                continue

            if duration_seconds < 600:
                snippet = details_items[0].get("snippet", {})
                return {
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "duration": duration_seconds,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
    return None

# -------------------------
# Gemini API Call
# -------------------------
def call_gemini_api(prompt, document_content):
    headers = {
        "Content-Type": "application/json"
    }
    max_doc_length = 30000  # Limit to avoid exceeding API limits
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
    print(f"URL: {GEMINI_API_URL.split('?')[0]}?key=...")  # Hiding the key in the log

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

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
                "key_points": [],
                "learning_objectives": [],
                "free_resources": []
            }

        try:
            if generated_text.strip().startswith("```json"):
                generated_text = generated_text.strip()[7:-3].strip()
            elif generated_text.strip().startswith("```"):
                generated_text = generated_text.strip()[3:-3].strip()

            parsed_output = json.loads(generated_text)
            title = parsed_output.get("title", "No title extracted")
            summary = parsed_output.get("summary", "No summary extracted")
            key_points = parsed_output.get("key_points", [])
            if not isinstance(key_points, list):
                print(f"Warning: 'key_points' from Gemini was not a list, using empty list. Value: {key_points}")
                key_points = []
            # Retrieve additional fields if they exist
            learning_objectives = parsed_output.get("learning_objectives", [])
            free_resources = parsed_output.get("free_resources", [])
            return {
                "title": title,
                "summary": summary,
                "key_points": key_points,
                "learning_objectives": learning_objectives,
                "free_resources": free_resources,
                "sub_topics": parsed_output.get("sub_topics", []),
                "mastering_plan": parsed_output.get("mastering_plan", [])
            }
        except json.JSONDecodeError:
            print("Warning: Gemini response was not valid JSON. Using the full text as summary.")
            return {
                "title": "Title Generation Failed (Non-JSON)",
                "summary": generated_text,
                "key_points": [],
                "learning_objectives": [],
                "free_resources": [],
                "sub_topics": [],
                "mastering_plan": []
            }

    except requests.exceptions.Timeout as e:
        print(f"Error calling Gemini API (Timeout): {e}")
        return {
            "title": "Error",
            "summary": f"API request timed out after 120 seconds: {e}",
            "key_points": [],
            "learning_objectives": [],
            "free_resources": [],
            "sub_topics": [],
            "mastering_plan": []
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
            "key_points": [],
            "learning_objectives": [],
            "free_resources": [],
            "sub_topics": [],
            "mastering_plan": []
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
            "key_points": [],
            "learning_objectives": [],
            "free_resources": [],
            "sub_topics": [],
            "mastering_plan": []
        }

# -------------------------
# Building the Knowledge Base
# -------------------------
def build_knowledge_base(query):
    search_results = google_search(query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID)
    knowledge_base = []
    if not search_results:
        print("No search results found for the query.")
        return []

    for item in search_results:
        link = item.get("link")
        title_from_search = item.get("title", "N/A")
        if not link:
            print("Search result item missing 'link'. Skipping.")
            continue

        print(f"\nProcessing: {link} ({title_from_search})")
        html_content = fetch_page_content(link)
        if not html_content:
            print(f"Skipping {link} due to fetch error.")
            continue

        text_content = extract_text_from_html(html_content)
        if not text_content or len(text_content) < 100:
            print(f"Skipping {link} due to insufficient text content ({len(text_content)} chars).")
            continue

        print(f"Extracted ~{len(text_content)} characters of text. Calling Gemini...")

        # Updated enhanced prompt with additional educational details and requirement to include YouTube links:
        prompt = f"""
Analyze the following document content related to the topic '{query}' and create an interactive learning roadmap designed to help a student master this subject. Your response must be strictly in JSON format with the following keys:

- "title": Provide a clear and concise title summarizing the main subject of the document.
- "summary": Write a detailed summary (at least 250 words) that covers the key insights, educational value, and main ideas found in the document.
- "sub_topics": Identify and list all relevant subtopics that are essential for mastering the topic. For each subtopic, include an object with:
    - "name": The name of the subtopic.
    - "description": A brief explanation of why this subtopic is important and how it relates to the main topic.
    - "free_resources": A list of free, popular resources (such as websites, articles, tutorials, videos, including at least one YouTube video link if available) that are specifically useful for learning this subtopic.
- "mastering_plan": Propose an interactive, step-by-step plan outlining how someone can progressively master the entire topic. This plan should integrate the subtopics and their corresponding free resources, and may include suggestions on interactive activities, self-assessment tips, or discussion prompts to deepen understanding.

If the document content is irrelevant to '{query}', too short, or appears to be an error page, return JSON with the following structure:
{{
  "title": "Extraction Failed",
  "summary": "Content irrelevant or insufficient",
  "sub_topics": [],
  "mastering_plan": []
}}

JSON Output:
"""
        structured_data = call_gemini_api(prompt, text_content)

        if structured_data.get("title") == "Error" or structured_data.get("title") == "Extraction Failed":
            print(f"Gemini extraction failed or content was irrelevant for {link}. Summary: {structured_data.get('summary')}")
            # Optionally skip adding failed extractions to the KB:
            # continue
        else:
            print(f"Successfully processed: Title - {structured_data.get('title')}")

        # New: Integrate YouTube video search for each subtopic (if YOUTUBE_API_KEY is set)
        if YOUTUBE_API_KEY and "sub_topics" in structured_data and isinstance(structured_data["sub_topics"], list):
            for subtopic in structured_data["sub_topics"]:
                subtopic_name = subtopic.get("name")
                if subtopic_name:
                    combined_query = f"{query} {subtopic_name}"
                    youtube_video = get_youtube_video_for_subtopic(combined_query, YOUTUBE_API_KEY)
                    if youtube_video:
                        subtopic.setdefault("free_resources", [])
                        subtopic["free_resources"].append({
                            "type": "youtube",
                            "title": youtube_video.get("title"),
                            "url": youtube_video.get("url"),
                            "description": youtube_video.get("description"),
                            "duration_seconds": youtube_video.get("duration")
                        })
                    else:
                        print(f"Warning: No suitable YouTube video found for subtopic '{subtopic_name}'.")

        # Append the source URL for traceability
        structured_data["source_url"] = link
        knowledge_base.append(structured_data)
        time.sleep(0.5)

    return knowledge_base

# -------------------------
# Embedding Generation (Using Sentence Transformers)
# -------------------------
def generate_embedding(text):
    """
    Generates an embedding for the given text using the Sentence Transformer model.
    Returns the embedding as a list of floats.
    """
    if not text or not isinstance(text, str):
        print("Warning: Received invalid text. Returning zero vector.")
        return [0.0] * EMBEDDING_DIM

    # Generate the embedding vector with no progress bar for efficiency.
    embedding = embedding_model.encode(text, show_progress_bar=False)
    return embedding.tolist()

# -------------------------
# Pinecone Integration
# -------------------------
def upsert_to_pinecone(knowledge_base, topic):
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
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Index '{INDEX_NAME}' creation initiated. Waiting for it to be ready...")
                while True:
                    try:
                        index_description = pc.describe_index(INDEX_NAME)
                        if index_description and index_description["status"]["ready"]:
                            print(f"Index '{INDEX_NAME}' is ready.")
                            break
                    except PineconeApiException as desc_e:
                        print(f"Waiting for index... (describe error: {desc_e})")
                    time.sleep(10)
            except PineconeApiException as create_e:
                print(f"Error creating Pinecone index: {create_e}")
                if "quota" in str(create_e).lower():
                    print("Suggestion: Check your Pinecone plan limits and existing indexes.")
                return
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
                print(f"Warning: Duplicate ID generated ({vector_id}), appending UUID.")
                vector_id = f"{vector_id}-{uuid.uuid4()}"
            ids_generated.add(vector_id)

            # Add extra metadata: topic and last_updated
            metadata = {
                "topic": topic,
                "last_updated": datetime.now().isoformat(),
                "title": str(item.get("title", "N/A"))[:500],
                "summary": str(text_to_embed)[:2000],
                "source_url": str(item.get("source_url", "N/A")),
                "key_points": json.dumps(item.get("key_points", [])),
                "learning_objectives": json.dumps(item.get("learning_objectives", [])),
                "free_resources": json.dumps(item.get("free_resources", []))
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
# Helper: Check for Existing Topic in Pinecone
# -------------------------
def topic_exists_recently(topic):
    """
    Queries the Pinecone index to see if any vectors with the specified topic exist
    and whether their last update timestamp is within the last 5 days.
    """
    try:
        index = pc.Index(INDEX_NAME)
        # Query with a metadata filter for the topic.
        query_response = index.query(
            filter={"topic": topic},
            top_k=1,  # Only need one match to check freshness.
            include_metadata=True
        )
        matches = query_response.get("matches", [])
        if matches:
            last_updated_iso = matches[0]["metadata"].get("last_updated")
            if last_updated_iso:
                last_updated = datetime.fromisoformat(last_updated_iso)
                if datetime.now() - last_updated < timedelta(days=5):
                    print("Topic exists and was updated less than 5 days ago.")
                    return True
        return False
    except Exception as e:
        print(f"Error querying Pinecone for existing topic: {e}")
        # In case of error, proceed with reprocessing.
        return False

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    topic = input("Enter the topic for knowledge base creation: ").strip()
    if not topic:
        print("Topic cannot be empty.")
    else:
        # Check if the topic exists and is updated within the last 5 days.
        if topic_exists_recently(topic):
            print("A recent knowledge base for this topic already exists in Pinecone. Skipping LLM extraction.")
        else:
            print(f"\n--- Starting Knowledge Base Creation for: {topic} ---")
            kb = build_knowledge_base(topic)

            print("\n--- Knowledge Base Creation Complete ---")
            print(f"Generated {len(kb)} entries for the knowledge base.")

            if kb:
                upsert_to_pinecone(kb, topic)
            else:
                print("No entries generated, skipping Pinecone upsert.")

        print("\n--- Script Finished ---")
