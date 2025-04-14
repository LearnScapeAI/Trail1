from dotenv import load_dotenv
import os
import sys
import requests
import json
import time
import re
import isodate
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, PineconeApiException

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = "knowledge-base-index"

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("Error: API keys not set.")
    sys.exit(1)

# -------------------------
# Initialize Pinecone and Embedding Model
# -------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"Error accessing Pinecone index '{INDEX_NAME}': {e}")
    sys.exit(1)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384

# -------------------------
# Utility Functions
# -------------------------
def clean_generated_text(text):
    """
    Removes common markdown formatting, such as starting and ending code block markers.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def call_gemini_api(prompt, document_content):
    """
    Calls the Gemini API with the concatenated prompt and document content, cleans the output,
    and returns the generated roadmap as a Python dictionary.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt + "\n\n" + document_content}]
        }],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8192}
    }
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean the generated text to remove any markdown formatting
        generated_text = clean_generated_text(generated_text)
        
        # Debug: Print cleaned generated text for inspection
        print("Cleaned generated text:")
        print(generated_text)
        
        try:
            roadmap = json.loads(generated_text)
        except json.JSONDecodeError as jde:
            print("JSON decode error:", jde)
            print("Raw generated text content for inspection:")
            print(generated_text)
            return None
        
        return roadmap
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_embedding(text):
    """
    Generates an embedding for the given text using the SentenceTransformer model.
    """
    return embedding_model.encode(text, show_progress_bar=False).tolist()

def query_knowledge_base(query, top_k=3):
    """
    Queries the Pinecone index using an embedding of the query text and returns
    the top_k matching documents.
    """
    query_embedding = generate_embedding(query)
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="default")
    return response.get("matches", [])

def generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration):
    """
    Constructs the interactive prompt for the Gemini API.
    This prompt instructs the API to generate an individual day-by-day roadmap (with one entry per day)
    that allocates tasks based on the daily time commitment without grouping days, ensuring that time dedicated
    to each subtopic is adequate.
    """
    prompt = (
        f"You're tasked with generating an actionable, detailed, and realistic day-by-day roadmap for mastering '{topic}'.\n"
        f"Skill level: '{skill_level}'. Daily commitment: '{daily_time_commitment}' (e.g., 2 hours per day). Duration: '{roadmap_duration}' days.\n\n"
        "Important Instructions:\n"
        f"- Provide exactly {roadmap_duration} individual entries, one for each day (e.g., 'Day 1', 'Day 2', ..., 'Day {roadmap_duration}'). Do not group multiple days together.\n"
        "- For every day, break down the learning tasks into clear segments that collectively add up exactly to the daily time commitment. "
        "Each subtopic should be assigned a realistic amount of time based on the overall daily commitment (for example, if the daily commitment is 2 hours, ensure that tasks split the 2 hours appropriately).\n"
        "- Ensure that the tasks are detailed and include specific learning goals, hands-on practice, and even short breaks or transitions if needed.\n"
        "- Incorporate useful resource links or references where it makes sense to enhance the learning experience. Also, include at least one relevant YouTube video link per subtopic if available.\n"
        "- The output must be provided strictly in JSON format, without any Markdown, code block indicators, or extra characters.\n\n"
        "Expected JSON structure:\n"
        "{\n"
        "  \"roadmap_title\": \"<Title of the Roadmap>\",\n"
        "  \"steps\": [\n"
        "    {\"Day 1\": [\n"
        "                {\"Task 1 (X hour)\": \"Task description and objectives.\"},\n"
        "                {\"Task 2 (Y hour)\": \"Task description and objectives.\"}\n"
        "              ]},\n"
        "    {\"Day 2\": [\n"
        "                {\"Task 1 (X hour)\": \"Task description and objectives.\"},\n"
        "                {\"Task 2 (Y hour)\": \"Task description and objectives.\"}\n"
        "              ]},\n"
        "    ... (continue for exactly " + str(roadmap_duration) + " days)\n"
        "  ]\n"
        "}\n"
    )
    return prompt

def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    """
    Given a subtopic query, search YouTube using the Data API and return the first video
    that meets the criteria (duration < 600 seconds). Returns a dictionary with relevant video info.
    """
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
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Gather user inputs for the roadmap
    topic = input("Enter topic: ").strip()
    skill_level = input("Skill level (Beginner/Intermediate/Advanced): ").strip()
    daily_time_commitment = input("Daily time commitment (e.g., '2 hours'): ").strip()
    roadmap_duration = int(input("Roadmap duration (days): ").strip())

    # Query Knowledge Base (e.g., to get relevant summaries from your documents)
    retrieved_docs = query_knowledge_base(topic)
    if not retrieved_docs:
        print(f"No relevant documents found for '{topic}'.")
        sys.exit(1)

    # Aggregate context summaries from retrieved documents
    context_text = "\n".join(
        doc["metadata"].get("summary", "")
        for doc in retrieved_docs if doc["metadata"].get("summary")
    )
    
    # Generate prompt with enhanced instructions
    prompt = generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration)
    
    # Call Gemini API to generate the roadmap based on the prompt and context
    roadmap = call_gemini_api(prompt, context_text)
    
    # Append YouTube video links (if available) to each day/task in the roadmap.
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    if YOUTUBE_API_KEY and roadmap and "steps" in roadmap:
        for day in roadmap["steps"]:
            # Each day is represented as a dictionary with a single key (e.g., 'Day 1').
            for day_key in day:
                tasks = day[day_key]
                # Loop over each task dictionary in the list for the day.
                for task in tasks:
                    for task_title, task_desc in list(task.items()):
                        # Use the task title and topic as the search query for YouTube.
                        youtube_video = get_youtube_video_for_subtopic(f"{topic} {task_title}", YOUTUBE_API_KEY)
                        if youtube_video:
                            # Append YouTube link to the task entry by adding a new key.
                            task[f"{task_title} - YouTube Link"] = youtube_video["url"]
                        else:
                            print(f"Warning: No suitable YouTube video found for task '{task_title}'.")
    
    if roadmap:
        print(json.dumps(roadmap, indent=2))
    else:
        print("Failed to generate roadmap.")
