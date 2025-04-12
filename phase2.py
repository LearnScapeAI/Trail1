from dotenv import load_dotenv
import os
import sys
import requests
import json
import time
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
# Functions
# -------------------------
def call_gemini_api(prompt, document_content):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt + "\n\n" + document_content}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8192}
    }
    try:
        response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:-3].strip()
        return json.loads(generated_text)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_embedding(text):
    return embedding_model.encode(text, show_progress_bar=False).tolist()

def query_knowledge_base(query, top_k=3):
    query_embedding = generate_embedding(query)
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="default")
    return response.get("matches", [])

def generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration):
    prompt = (
        f"You're tasked with generating an actionable, detailed, and realistic day-by-day roadmap for mastering '{topic}'.\n"
        f"Skill level: '{skill_level}'. Daily commitment: '{daily_time_commitment}'. Duration: '{roadmap_duration}' days.\n\n"
        "Structure a practical, sequential plan starting from Day 1. Include specific tasks, milestones, and actionable steps, integrating useful resource links from the context.\n\n"
        f"Context:\n{context_text}\n\n"
        "Format response in JSON:\n- 'roadmap_title': title\n- 'steps': list detailing each day (e.g., 'Day 1:...')"
    )
    return prompt

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    topic = input("Enter topic: ").strip()
    skill_level = input("Skill level (Beginner/Intermediate/Advanced): ").strip()
    daily_time_commitment = input("Daily time commitment (e.g., '2 hours'): ").strip()
    roadmap_duration = int(input("Roadmap duration (days): ").strip())

    retrieved_docs = query_knowledge_base(topic)
    if not retrieved_docs:
        print(f"No relevant documents found for '{topic}'.")
        sys.exit(1)

    context_text = "\n".join(doc["metadata"].get("summary", "") for doc in retrieved_docs if doc["metadata"].get("summary"))

    prompt = generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration)
    roadmap = call_gemini_api(prompt, context_text)

    if roadmap:
        print(json.dumps(roadmap, indent=2))
    else:
        print("Failed to generate roadmap.")
