# -------------------------
# gemini_utils.py
# -------------------------
import logging
logger = logging.getLogger(__name__)

def clean_generated_text(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def call_gemini_api(prompt, document_content, api_key):
    import requests, json
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt + "\n\n" + document_content}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8192}
    }
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        generated_text = clean_generated_text(generated_text)
        logger.debug("Cleaned Gemini response:\n%s", generated_text)
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError as jde:
            logger.error("JSON decode error: %s", jde)
            return None
    except Exception as e:
        logger.exception("Gemini API error")
        return None

# -------------------------
# pinecone_utils.py
# -------------------------
import logging
logger = logging.getLogger(__name__)

def generate_embedding(text, model):
    return model.encode(text, show_progress_bar=False).tolist()

def query_knowledge_base(index, query, model, top_k=5):
    embedding = generate_embedding(query, model)
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace="default")
    matches = response.get("matches", [])
    if not matches:
        logger.warning("No matching documents in Pinecone for query: '%s'", query)
    return matches

# -------------------------
# youtube_utils.py
# -------------------------
import logging
logger = logging.getLogger(__name__)

def get_youtube_video_for_subtopic(subtopic_query, api_key, max_results=5):
    import requests, isodate
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
        logger.warning("YouTube search error: %s", e)
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
            logger.debug("Error fetching video details: %s", e)
            continue
    return None

# -------------------------
# main.py (entrypoint)
# -------------------------
from dotenv import load_dotenv
import os
import sys
import json
import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from gemini_utils import call_gemini_api
from pinecone_utils import query_knowledge_base
from youtube_utils import get_youtube_video_for_subtopic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
INDEX_NAME = "knowledge-base-index"

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.error("Missing API keys.")
    sys.exit(1)

# Init
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Inputs
topic = input("Enter topic: ").strip()
skill_level = input("Skill level (Beginner/Intermediate/Advanced): ").strip() or "Intermediate"
daily_time = input("Daily time commitment (e.g., '2 hours'): ").strip() or "2 hours"
try:
    duration = int(input("Roadmap duration (days): ").strip())
except ValueError:
    logger.warning("Invalid duration input. Using default 30 days.")
    duration = 30

# Prompt Builder
def generate_prompt(topic, context, skill, time, days):
    return (
        f"You're tasked with generating a detailed day-by-day roadmap for mastering '{topic}' for a {skill} learner.\n"
        f"Time: {time}/day, Duration: {days} days.\n\n"
        "Requirements:\n"
        f"- Return exactly {days} day entries.\n"
        "- Each day must have specific tasks summing up to the daily time.\n"
        "- Include clear goals, short breaks if needed, and relevant YouTube links.\n"
        "- Format must be JSON only (no markdown).\n\n"
        "Expected format:\n"
        "{\"roadmap_title\": \"...\", \"steps\": [{\"Day 1\": [{\"Task 1\": \"...\"}, ...]}, ...]}"
    )

# Query KB
matches = query_knowledge_base(index, topic, embedding_model)
if not matches:
    logger.warning("No documents found in Pinecone for topic '%s'", topic)
    sys.exit(1)

context = "\n".join(m["metadata"].get("summary", "") for m in matches)
prompt = generate_prompt(topic, context, skill_level, daily_time, duration)
logger.info("Calling Gemini API for roadmap generation...")
roadmap = call_gemini_api(prompt, context, GEMINI_API_KEY)

if YOUTUBE_API_KEY and roadmap and "steps" in roadmap:
    logger.info("Enriching roadmap with YouTube video links...")
    for day in roadmap["steps"]:
        for _, tasks in day.items():
            for task in tasks:
                for title in list(task):
                    yt = get_youtube_video_for_subtopic(f"{topic} {title}", YOUTUBE_API_KEY)
                    if yt:
                        task[f"{title} - YouTube Link"] = yt["url"]

if roadmap:
    logger.info("Final roadmap:")
    print(json.dumps(roadmap, indent=2))
else:
    logger.error("Roadmap generation failed.")
