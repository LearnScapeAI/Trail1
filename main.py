from dotenv import load_dotenv
import os
import sys
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from gemini_utils import call_gemini_api
from pinecone_utils import query_knowledge_base
from youtube_utils import get_youtube_video_for_subtopic

# Load Environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
INDEX_NAME = "knowledge-base-index"

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("Missing API keys.")
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
    print("No documents found.")
    sys.exit(1)

context = "\n".join(m["metadata"].get("summary", "") for m in matches)
prompt = generate_prompt(topic, context, skill_level, daily_time, duration)
roadmap = call_gemini_api(prompt, context, GEMINI_API_KEY)

if YOUTUBE_API_KEY and roadmap and "steps" in roadmap:
    for day in roadmap["steps"]:
        for _, tasks in day.items():
            for task in tasks:
                for title in list(task):
                    yt = get_youtube_video_for_subtopic(f"{topic} {title}", YOUTUBE_API_KEY)
                    if yt:
                        task[f"{title} - YouTube Link"] = yt["url"]

print(json.dumps(roadmap, indent=2) if roadmap else "Failed to generate roadmap.")
