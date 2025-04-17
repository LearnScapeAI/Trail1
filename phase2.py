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
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def call_gemini_api(prompt, document_content):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt + "\n\n" + document_content}]}],
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
        generated_text = clean_generated_text(generated_text)
        print("Cleaned generated text:")
        print(generated_text)
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError as jde:
            print("JSON decode error:", jde)
            return None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_embedding(text):
    return embedding_model.encode(text, show_progress_bar=False).tolist()

def query_knowledge_base(query, top_k=5):
    query_embedding = generate_embedding(query)
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="default")
    return response.get("matches", [])

def generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration):
    return (
        f"You're tasked with generating a detailed day-by-day roadmap for mastering '{topic}' for a {skill_level} learner.\n"
        f"Time: {daily_time_commitment}/day, Duration: {roadmap_duration} days.\n\n"
        "Requirements:\n"
        f"- Return exactly {roadmap_duration} day entries.\n"
        "- Each day must have specific tasks summing up to the daily time.\n"
        "- Include clear goals, short breaks if needed, and relevant YouTube links.\n"
        "- Format must be JSON only (no markdown).\n\n"
        "Expected format:\n"
        "{\"roadmap_title\": \"...\", \"steps\": [{\"Day 1\": [{\"Task 1\": \"...\"}, ...]}, ...]}"
    )

def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    try:
        response = requests.get(search_url, params={
            "part": "snippet", "q": subtopic_query, "key": api_key,
            "maxResults": max_results, "type": "video"
        })
        response.raise_for_status()
        results = response.json().get("items", [])
    except Exception as e:
        print(f"YouTube search error: {e}")
        return None

    for item in results:
        video_id = item["id"]["videoId"]
        try:
            details = requests.get(video_url, params={
                "part": "contentDetails,snippet", "id": video_id, "key": api_key
            }).json()
            duration = isodate.parse_duration(
                details["items"][0]["contentDetails"]["duration"]
            ).total_seconds()
            if duration < 600:
                snippet = details["items"][0]["snippet"]
                return {
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "duration": duration,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
        except Exception as e:
            print(f"Video details error: {e}")
            continue
    return None

if __name__ == "__main__":
    topic = input("Enter topic: ").strip()
    skill_level = input("Skill level (Beginner/Intermediate/Advanced): ").strip() or "Intermediate"
    daily_time_commitment = input("Daily time commitment (e.g., '2 hours'): ").strip() or "2 hours"
    try:
        roadmap_duration = int(input("Roadmap duration (days): ").strip())
    except ValueError:
        roadmap_duration = 30

    docs = query_knowledge_base(topic)
    if not docs:
        print("No documents found.")
        sys.exit(1)

    context_text = "\n".join(doc["metadata"].get("summary", "") for doc in docs)
    prompt = generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration)
    roadmap = call_gemini_api(prompt, context_text)

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if YOUTUBE_API_KEY and roadmap and "steps" in roadmap:
        for day in roadmap["steps"]:
            for day_key, tasks in day.items():
                for task in tasks:
                    for task_title in list(task):
                        video = get_youtube_video_for_subtopic(f"{topic} {task_title}", YOUTUBE_API_KEY)
                        if video:
                            task[f"{task_title} - YouTube Link"] = video["url"]

    print(json.dumps(roadmap, indent=2) if roadmap else "Failed to generate roadmap.")
