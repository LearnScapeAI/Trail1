from dotenv import load_dotenv
import os
import sys
import requests
import json
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, PineconeApiException  # Using new Pinecone Client

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')  # Optional: Add your Pinecone environment if needed
INDEX_NAME = "knowledge-base-index"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not set.")
    sys.exit(1)
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not set.")
    sys.exit(1)
if not PINECONE_ENVIRONMENT:
    print("Warning: PINECONE_ENVIRONMENT not set. Assuming default environment.")

# -------------------------
# Gemini API Configuration
# -------------------------
MODEL_NAME = "gemini-1.5-pro-latest"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# -------------------------
# Initialize Pinecone and Embedding Model
# -------------------------
try:
    # Initialize Pinecone client using the provided API key.
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Retrieve the index; this assumes the index already exists.
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"Error accessing Pinecone index '{INDEX_NAME}': {e}")
    sys.exit(1)

# Load an embedding model (e.g., 'all-MiniLM-L6-v2' outputs 384-dimensional vectors)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384

# -------------------------
# Gemini API Call Function
# -------------------------
def call_gemini_api(prompt, document_content):
    headers = {"Content-Type": "application/json"}
    max_doc_length = 30000
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
    print(f"URL: {GEMINI_API_URL.split('?')[0]}?key=...")  # Hide API key in logs
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        try:
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as parse_err:
            error_info = f"Could not extract generated text from API response. Error: {parse_err}."
            prompt_feedback = result.get("promptFeedback")
            if prompt_feedback:
                block_reason = prompt_feedback.get("blockReason")
                safety_ratings = prompt_feedback.get("safetyRatings")
                error_info += f" Prompt Feedback: BlockReason={block_reason}, SafetyRatings={safety_ratings}"
            print("Warning:", error_info)
            return {"roadmap_title": "Error", "steps": []}
        if generated_text.strip().startswith("```json"):
            generated_text = generated_text.strip()[7:-3].strip()
        elif generated_text.strip().startswith("```"):
            generated_text = generated_text.strip()[3:-3].strip()
        try:
            parsed_output = json.loads(generated_text)
            return parsed_output
        except json.JSONDecodeError:
            print("Warning: Gemini response was not valid JSON. Using full text as output.")
            return {"roadmap_title": f"Generation Failed (Non-JSON)", "steps": []}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"roadmap_title": "Error", "steps": []}

# -------------------------
# Embedding Generation Function
# -------------------------
def generate_embedding(text):
    if not text or not isinstance(text, str):
        print("Warning: generate_embedding received invalid text. Returning zero vector.")
        return [0.0] * EMBEDDING_DIM
    embedding = embedding_model.encode(text, show_progress_bar=False)
    return embedding.tolist()

# -------------------------
# Pinecone Query Function
# -------------------------
def query_knowledge_base(query, top_k=3):
    query_embedding = generate_embedding(query)
    try:
        # Include the namespace "default" if data was stored in that namespace.
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"  # Make sure this matches your upsert namespace
        )
        return query_response.get("matches", [])
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# -------------------------
# Roadmap Generation Function
# -------------------------
def generate_roadmap(topic, top_k=3):
    print(f"--- Querying Pinecone for context on '{topic}' ---")
    retrieved_docs = query_knowledge_base(topic, top_k=top_k)
    if not retrieved_docs:
        print(f"No relevant documents found in the knowledge base for the topic '{topic}'. "
              "Please ensure the topic is well-represented in your data.")
        return None

    # Combine the 'summary' field from retrieved documents to form the context.
    context_parts = []
    for doc in retrieved_docs:
        summary = doc.get("metadata", {}).get("summary", "")
        if summary:
            context_parts.append(summary)
    context_text = "\n".join(context_parts)
    if context_text:
        print("Retrieved context (first 500 characters):")
        print(context_text[:500] + ("..." if len(context_text) > 500 else ""))
    else:
        print("No 'summary' field found in the metadata of the retrieved documents.")

    prompt = (
    f"Based on the following information, create an actionable, realistic, and detailed day-by-day roadmap for the topic '{topic}'. "
    "The roadmap should provide a step-by-step plan starting from Day 1, outlining specific tasks, milestones, and actionable items that a person can realistically follow. "
    "For each day, detail what needs to be accomplished, ensuring the plan is practical and sequential. "
    "If there are any useful resource links present in the context from the knowledge base, include them in the appropriate steps to support further learning and implementation. "
    "Synthesize the roadmap using the details provided in the context below.\n\n"
    "Context:\n"
    f"{context_text}\n\n"
    "Format the output strictly in JSON with the keys 'roadmap_title' and 'steps', where 'steps' is a list of bullet points that detail the day-by-day plan (e.g., 'Day 1: ...', 'Day 2: ...') along with any resource links if available."
)

    roadmap = call_gemini_api(prompt, context_text)
    return roadmap

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    topic = input("Enter the topic for roadmap generation: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        sys.exit(1)
    print(f"--- Generating Roadmap for '{topic}' ---")
    roadmap = generate_roadmap(topic)
    if roadmap:
        print("\n--- Generated Roadmap ---")
        print(json.dumps(roadmap, indent=2))
    else:
        print("Failed to generate a roadmap.")
