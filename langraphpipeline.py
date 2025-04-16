import json
import sys

# -------------------------------------------------------
# Minimal Node and Graph Classes for a Simple Pipeline
# -------------------------------------------------------

class Node:
    def run(self, inputs: dict) -> dict:
        """
        Override this method in subclasses to process input data
        and return a dictionary as output.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

class Graph:
    def __init__(self):
        self.nodes = {}
        self.order = []  # sequential order of nodes

    def add_node(self, name, node):
        self.nodes[name] = node
        self.order.append(name)

    def add_edge(self, from_node, to_node):
        # In this simple version, edges are not explicitly used.
        pass

    def run(self, initial_input):
        result = initial_input
        for name in self.order:
            node = self.nodes[name]
            result = node.run(result)
        return result

# -------------------------------------------------------
# Import Your Functions from Other Modules
# -------------------------------------------------------

# Ensure your module files are in the same directory.
try:
    from knowledgecollection import build_knowledge_base, upsert_to_pinecone
except ImportError:
    from KNOWLEDGECOLLECTION import build_knowledge_base, upsert_to_pinecone

try:
    from phase2 import query_knowledge_base, generate_interactive_prompt, call_gemini_api
except ImportError:
    print("Error: Could not import functions from phase2.py")
    sys.exit(1)

# -------------------------------------------------------
# Node Definitions for the Pipeline
# -------------------------------------------------------

class KnowledgeCollectionNode(Node):
    def run(self, inputs: dict) -> dict:
        """
        This node builds the knowledge base for the given topic and upserts the results to Pinecone.
        It expects the input dictionary to contain at least a key "topic".
        After processing, it returns the entire input so that additional parameters are preserved.
        """
        topic = inputs.get("topic")
        if not topic:
            raise ValueError("Topic must be provided in the input.")

        print(f"[KnowledgeCollectionNode] Building knowledge base for: {topic}")
        kb = build_knowledge_base(topic)
        upsert_to_pinecone(kb, topic)
        # Return the original input, so additional parameters (skill level, etc.) pass along.
        return inputs

class RoadmapGenerationNode(Node):
    def run(self, inputs: dict) -> dict:
        """
        This node queries the Pinecone knowledge base, aggregates document summaries,
        constructs a prompt, and calls the Gemini API to generate a detailed learning roadmap.
        It reads additional parameters from the input.
        """
        topic = inputs.get("topic")
        print(f"[RoadmapGenerationNode] Generating roadmap for: {topic}")
        
        retrieved_docs = query_knowledge_base(topic)
        if not retrieved_docs:
            raise ValueError("No documents found in Pinecone for this topic.")

        # Aggregate the summaries into a single context string.
        context_text = "\n".join(
            doc.get("metadata", {}).get("summary", "")
            for doc in retrieved_docs if doc.get("metadata", {}).get("summary")
        )

        # Read additional parameters from input, with defaults.
        skill_level = inputs.get("skill_level", "Intermediate")
        daily_time_commitment = inputs.get("daily_time_commitment", "2 hours")
        try:
            roadmap_duration = int(inputs.get("roadmap_duration", 30))
        except ValueError:
            roadmap_duration = 30

        # Generate the interactive prompt.
        prompt = generate_interactive_prompt(topic, context_text, skill_level, daily_time_commitment, roadmap_duration)
        
        # Call the Gemini API to generate the roadmap.
        roadmap = call_gemini_api(prompt, context_text)
        return {"roadmap": roadmap, **inputs}

# -------------------------------------------------------
# Graph Construction and Execution with User Prompts
# -------------------------------------------------------

def main():
    # Prompt the user for input parameters.
    topic = input("Enter the topic: ").strip()
    while not topic:
        print("Topic cannot be empty.")
        topic = input("Enter the topic: ").strip()
    
    skill_level = input("Enter skill level (Beginner/Intermediate/Advanced): ").strip()
    if not skill_level:
        skill_level = "Intermediate"
    
    daily_time_commitment = input("Enter daily time commitment (e.g., '2 hours'): ").strip()
    if not daily_time_commitment:
        daily_time_commitment = "2 hours"
    
    roadmap_duration_input = input("Enter roadmap duration (in days): ").strip()
    try:
        roadmap_duration = int(roadmap_duration_input)
    except ValueError:
        print("Invalid duration. Using default of 30 days.")
        roadmap_duration = 30

    # Build the initial input dictionary.
    initial_input = {
        "topic": topic,
        "skill_level": skill_level,
        "daily_time_commitment": daily_time_commitment,
        "roadmap_duration": roadmap_duration
    }

    # Create a new Graph pipeline.
    graph = Graph()
    graph.add_node("knowledge_collection", KnowledgeCollectionNode())
    graph.add_node("roadmap_generation", RoadmapGenerationNode())

    # In our simple pipeline, nodes execute sequentially.
    graph.add_edge("knowledge_collection", "roadmap_generation")

    # Run the graph pipeline with the initial user input.
    result = graph.run(initial_input)

    # Extract and print the final roadmap.
    roadmap = result.get("roadmap")
    if roadmap:
        print("Final Roadmap Output:")
        print(json.dumps(roadmap, indent=2))
    else:
        print("Roadmap generation failed.")

if __name__ == "__main__":
    main()
