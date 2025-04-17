from dotenv import load_dotenv
import os
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
import asyncio
import aiohttp
import isodate

from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from sentence_transformers import SentenceTransformer

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# Load API Keys from Environment Variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', "your_default_pinecone_key")
CUSTOM_SEARCH_ENGINE_ID = os.getenv('CUSTOM_SEARCH_ENGINE_ID', '627f547ffe4c94a8d')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')  # Optional: YouTube integration

# Validate required keys
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

INDEX_NAME = "knowledge-base-index"
EMBEDDING_DIM = 384  # Must match your Sentence Transformer model

# -------------------------
# Initialize Pinecone and Sentence Transformer
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Gemini API configuration
# -------------------------
MODEL_NAME = "gemini-1.5-pro-latest"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# -------------------------
# Synchronous helper: Google Custom Search
# -------------------------
def google_search(query, api_key, cse_id, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cse_id, 'q': query, 'num': num_results}
    try:
        import requests
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        return results.get("items", [])
    except Exception as e:
        print(f"Error during Google Search API call: {e}")
        return []

# -------------------------
# Synchronous helper: HTML text extraction
# -------------------------
def extract_text_from_html(html_content):
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.extract()
        text = soup.get_text(separator=" ", strip=True)
        return ' '.join(text.split())
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""

# -------------------------
# Synchronous helper: YouTube Search with 403 Handling and Fallback
# -------------------------
def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    """
    Given a subtopic query, search YouTube using the Data API and return the first video
    (duration < 600 sec) that meets the criteria. If no video is found for the combined query,
    try with the subtopic name alone. Specifically handles 403 errors.
    Returns a dictionary with video info if found; otherwise, returns None.
    """
    import requests
    import isodate
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    def search_youtube(query):
        params = {
            "part": "snippet",
            "q": query,
            "key": api_key,
            "maxResults": max_results,
            "type": "video"
        }
        try:
            search_response = requests.get(search_url, params=params)
            search_response.raise_for_status()
            return search_response.json().get("items", [])
        except requests.exceptions.HTTPError as he:
            if he.response.status_code == 403:
                print(f"Error: Received 403 Forbidden for YouTube search with query '{query}'. Please check your API key restrictions.")
            else:
                print(f"Error during YouTube search API call: {he}")
            return None
        except Exception as e:
            print(f"Error during YouTube search API call: {e}")
            return None

    results = search_youtube(subtopic_query)
    if not results:
        # Fallback: try using only the subtopic (remove topic prefix if exists)
        parts = subtopic_query.split()
        if len(parts) > 1:
            fallback_query = " ".join(parts[1:])
            print(f"Fallback: Trying YouTube search with subtopic '{fallback_query}'")
            results = search_youtube(fallback_query)
        if not results:
            return None

    for item in results:
        video_id = item["id"]["videoId"]
        params_details = {"part": "contentDetails,snippet", "id": video_id, "key": api_key}
        try:
            details_response = requests.get(video_url, params=params_details)
            details_response.raise_for_status()
            details_items = details_response.json().get("items", [])
        except requests.exceptions.HTTPError as he:
            if he.response.status_code == 403:
                print(f"Error: Received 403 Forbidden for YouTube video details for video '{video_id}'. Check your API key restrictions.")
            else:
                print(f"Error retrieving YouTube video details: {he}")
            continue
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
# Asynchronous helper: Fetch page content with aiohttp
# -------------------------
async def async_fetch_page_content(session, url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) ' +
                      'Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Failed to fetch {url} (status code {response.status})")
                return ""
    except asyncio.TimeoutError:
        print(f"Timeout error fetching {url}")
        return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# -------------------------
# Asynchronous helper: Call Gemini API with aiohttp
# -------------------------
async def async_call_gemini_api(prompt, document_content):
    headers = {"Content-Type": "application/json"}
    max_doc_length = 30000
    if len(document_content) > max_doc_length:
        print(f"Warning: Document content truncated to {max_doc_length} characters for Gemini API call.")
        document_content = document_content[:max_doc_length]
    combined_text = f"Prompt: {prompt}\n\nDocument:\n{document_content}"
    payload = {
        "contents": [{"parts": [{"text": combined_text}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8192}
    }
    print("--- Calling Gemini API ---")
    print(f"URL: {GEMINI_API_URL.split('?')[0]}?key=...")  # Hiding API key in logs
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120) as response:
                resp_json = await response.json()
                try:
                    generated_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError) as parse_err:
                    error_info = f"Could not extract generated text. Error: {parse_err}."
                    prompt_feedback = resp_json.get("promptFeedback")
                    if prompt_feedback:
                        block_reason = prompt_feedback.get("blockReason")
                        safety_ratings = prompt_feedback.get("safetyRatings")
                        error_info += f" Prompt Feedback: BlockReason={block_reason}, SafetyRatings={safety_ratings}"
                    else:
                        candidates = resp_json.get("candidates", [])
                        if not candidates:
                            error_info += " 'candidates' list is empty or missing."
                        elif candidates[0].get("finishReason") not in [None, "STOP", "MAX_TOKENS"]:
                            error_info += f" Candidate finishReason: {candidates[0].get('finishReason')}"
                    print("Warning:", error_info)
                    print("Full Response:", json.dumps(resp_json, indent=2))
                    return {"title": "Error", "summary": error_info, "key_points": [], "learning_objectives": [], "free_resources": []}
                generated_text = generated_text.strip()
                if generated_text.startswith("```json"):
                    generated_text = generated_text[len("```json"):].strip()
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3].strip()
                try:
                    parsed_output = json.loads(generated_text)
                    title = parsed_output.get("title", "No title extracted")
                    summary = parsed_output.get("summary", "No summary extracted")
                    key_points = parsed_output.get("key_points", [])
                    if not isinstance(key_points, list):
                        print("Warning: 'key_points' is not a list.")
                        key_points = []
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
                    print("Warning: Gemini response was not valid JSON. Using full text as summary.")
                    return {
                        "title": "Title Generation Failed (Non-JSON)",
                        "summary": generated_text,
                        "key_points": [],
                        "learning_objectives": [],
                        "free_resources": [],
                        "sub_topics": [],
                        "mastering_plan": []
                    }
    except asyncio.TimeoutError as e:
        print(f"Error calling Gemini API (Timeout): {e}")
        return {"title": "Error", "summary": f"API request timed out: {e}", "key_points": [], "learning_objectives": [], "free_resources": [], "sub_topics": [], "mastering_plan": []}
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e.__class__.__name__} - {e}")
        return {"title": "Error", "summary": f"An unexpected error occurred: {e}", "key_points": [], "learning_objectives": [], "free_resources": [], "sub_topics": [], "mastering_plan": []}

# -------------------------
# Asynchronous Knowledge Base Builder
# -------------------------
async def async_build_knowledge_base(query):
    search_results = google_search(query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID)
    knowledge_base = []
    if not search_results:
        print("No search results found for the query.")
        return []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        valid_results = []
        for item in search_results:
            link = item.get("link")
            if not link:
                print("Search result missing 'link'. Skipping.")
                continue
            tasks.append(async_fetch_page_content(session, link))
            valid_results.append(item)
        pages = await asyncio.gather(*tasks)
    
    for idx, page in enumerate(pages):
        if not page or len(page) < 100:
            print(f"Skipping {valid_results[idx].get('link')} (insufficient content, length={len(page) if page else 0}).")
            continue
        print(f"Fetched content from {valid_results[idx].get('link')} (length ~{len(page)} characters).")
        text_content = extract_text_from_html(page)
        if not text_content or len(text_content) < 100:
            print(f"Skipping {valid_results[idx].get('link')} after extraction (length={len(text_content) if text_content else 0}).")
            continue
        print(f"Extracted ~{len(text_content)} characters. Calling Gemini API asynchronously...")
        
        # Updated prompt to remove expectation of resource links
        prompt = f"""
Analyze the following document content related to the topic '{query}' and generate a structured JSON output with the following keys:
- "title": A concise title summarizing the document's main subject.
- "summary": A detailed summary (at least 250 words) covering key insights and main ideas.
- "sub_topics": An array of objects, each representing a subtopic essential for mastering the topic. Each object must include:
    - "name": The name of the subtopic.
    - "description": A brief explanation of why this subtopic is important and how it relates to the main topic.
- "mastering_plan": A step-by-step plan outlining how someone can progressively master the topic.
Return the output strictly as JSON with no markdown formatting.
"""
        structured_data = await async_call_gemini_api(prompt, text_content)
    
        if structured_data.get("title") in ["Error", "Extraction Failed"]:
            print(f"Gemini extraction failed for {valid_results[idx].get('link')}. Summary: {structured_data.get('summary')}")
            continue
        else:
            print(f"Successfully processed: Title - {structured_data.get('title')}")
        
        # YouTube enrichment: update subtopics with actual video links
        if YOUTUBE_API_KEY and "sub_topics" in structured_data and isinstance(structured_data["sub_topics"], list):
            for subtopic in structured_data["sub_topics"]:
                subtopic_name = subtopic.get("name")
                if subtopic_name:
                    combined_query = f"{query} {subtopic_name}"
                    youtube_video = get_youtube_video_for_subtopic(combined_query, YOUTUBE_API_KEY)
                    if not youtube_video:
                        print(f"Fallback: Trying YouTube search with subtopic '{subtopic_name}' only.")
                        youtube_video = get_youtube_video_for_subtopic(subtopic_name, YOUTUBE_API_KEY)
                    if youtube_video:
                        subtopic["free_resources"] = [{
                            "type": "youtube",
                            "title": youtube_video.get("title"),
                            "url": youtube_video.get("url"),
                            "description": youtube_video.get("description"),
                            "duration_seconds": youtube_video.get("duration")
                        }]
                    else:
                        print(f"Warning: No suitable YouTube video found for subtopic '{subtopic_name}'.")
    
        structured_data["source_url"] = valid_results[idx].get("link")
        knowledge_base.append(structured_data)
        await asyncio.sleep(0.5)
    
    return knowledge_base

# -------------------------
# Synchronous helper: Embedding Generation
# -------------------------
def generate_embedding(text):
    if not text or not isinstance(text, str):
        print("Warning: Received invalid text. Returning zero vector.")
        return [0.0] * EMBEDDING_DIM
    embedding = embedding_model.encode(text, show_progress_bar=False)
    return embedding.tolist()

# -------------------------
# Synchronous helper: Pinecone Upsert (wrapped in asyncio)
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
                        print(f"Waiting for index... (error: {desc_e})")
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
                print(f"Error upserting batch {i // batch_size + 1}: {upsert_e}")
                print(f"Problematic batch (first item ID): {batch[0]['id'] if batch else 'N/A'}")
                break
            except Exception as e:
                print(f"Unexpected error during batch upsert: {e}")
                break
        print("Pinecone upsert process completed.")

    except PineconeApiException as e:
        print(f"A Pinecone API error occurred: {e}")
        if "not found" in str(e).lower() and "index" in str(e).lower():
            print(f"Error: Index '{INDEX_NAME}' not found or access denied.")
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
    try:
        index = pc.Index(INDEX_NAME)
        query_response = index.query(
            filter={"topic": topic},
            top_k=1,
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
        print(f"Error querying Pinecone for topic: {e}")
        return False

# -------------------------
# Main Execution (Asynchronous)
# -------------------------
async def main():
    topic = input("Enter the topic for knowledge base creation: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return
    if topic_exists_recently(topic):
        print("A recent knowledge base for this topic already exists in Pinecone. Skipping LLM extraction.")
    else:
        print(f"\n--- Starting Knowledge Base Creation for: {topic} ---")
        kb = await async_build_knowledge_base(topic)
        print("\n--- Knowledge Base Creation Complete ---")
        print(f"Generated {len(kb)} entries for the knowledge base.")
        if kb:
            await asyncio.to_thread(upsert_to_pinecone, kb, topic)
        else:
            print("No entries generated, skipping Pinecone upsert.")
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
