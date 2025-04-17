import json
import sys
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from gemini_utils import call_gemini_api
from pinecone_utils import query_knowledge_base
from youtube_utils import get_youtube_video_for_subtopic

# -------------------------
# Minimal Node and Graph Classes
# -------------------------
class Node:
    def run(self, inputs: dict) -> dict:
        raise NotImplementedError("Subclasses must implement the run method.")

class Graph:
    def __init__(self):
        self.nodes = {}
        self.order = []

    def add_node(self, name, node):
        self.nodes[name] = node
        self.order.append(name)

    def add_edge(self, from_node, to_node):
        pass  # Not needed for sequential graph

    def run(self, initial_input):
        result = initial_input
        for name in self.order:
            result = self.nodes[name].run(result)
        return result

# -------------------------
# Node Implementations
# -------------------------
class RoadmapGenerationNode(Node):
    def run(self, inputs: dict) -> dict:
        topic = inputs.get("topic")
        skill_level = inputs.get("skill_level", "Intermediate")
        daily_time_commitment = inputs.get("daily_time_commitment", "2 hours")
        roadmap_duration = int(inputs.get("roadmap_duration", 30))

        matches = query_knowledge_base(inputs["index"], topic, inputs["embedding_model"])
        if not matches:
            raise ValueError(f"No documents found in Pinecone for topic '{topic}'")

        context = "\n".join(m["metadata"].get("summary", "") for m in matches)

        prompt = (
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

        roadmap = call_gemini_api(prompt, context, inputs["gemini_api_key"])

        if inputs.get("youtube_api_key") and roadmap and "steps" in roadmap:
            for day in roadmap["steps"]:
                for _, tasks in day.items():
                    for task in tasks:
                        for title in list(task):
                            yt = get_youtube_video_for_subtopic(f"{topic} {title}", inputs["youtube_api_key"])
                            if yt:
                                task[f"{title} - YouTube Link"] = yt["url"]

        inputs["roadmap"] = roadmap
        return inputs

# -------------------------
# Main Pipeline Execution
# -------------------------
def main():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    INDEX_NAME = "knowledge-base-index"

    if not GEMINI_API_KEY or not PINECONE_API_KEY:
        print("Missing API keys.")
        sys.exit(1)

    topic = input("Enter topic: ").strip()
    skill_level = input("Skill level (Beginner/Intermediate/Advanced): ").strip() or "Intermediate"
    daily_time = input("Daily time commitment (e.g., '2 hours'): ").strip() or "2 hours"
    try:
        duration = int(input("Roadmap duration (days): ").strip())
    except ValueError:
        duration = 30

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    initial_input = {
        "topic": topic,
        "skill_level": skill_level,
        "daily_time_commitment": daily_time,
        "roadmap_duration": duration,
        "gemini_api_key": GEMINI_API_KEY,
        "youtube_api_key": YOUTUBE_API_KEY,
        "embedding_model": embedding_model,
        "index": index
    }

    graph = Graph()
    graph.add_node("roadmap_generation", RoadmapGenerationNode())

    result = graph.run(initial_input)
    roadmap = result.get("roadmap")
    if roadmap:
        print(json.dumps(roadmap, indent=2))
    else:
        print("Roadmap generation failed.")

if __name__ == "__main__":
    main()
