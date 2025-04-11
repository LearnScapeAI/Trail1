# Retrieval-Augmented Generation (RAG) Educational Roadmap Generator

This project implements a two-part Retrieval-Augmented Generation (RAG) system that builds an educational knowledge base from web content and then generates personalized, day-by-day learning roadmaps. It combines vector-based semantic search, document summarization, and generative modeling to produce actionable learning plans.

## Features

- **Knowledge Base Construction**
  - Uses the Google Custom Search API to retrieve relevant web pages.
  - Scrapes and processes content with BeautifulSoup.
  - Calls the Gemini API to extract educational insights and structure them into JSON.
  - Generates embeddings via Sentence Transformers and upserts data into a Pinecone index.

- **Roadmap Generation**
  - Retrieves context by querying the Pinecone index using vector search.
  - Aggregates document summaries to form a relevant context.
  - Uses the Gemini API to generate a detailed, day-by-day educational roadmap based on the retrieved context.

## Tech Stack

- **Programming Language:** Python 3.x
- **Environment Management:** [python-dotenv](https://pypi.org/project/python-dotenv/)
- **HTTP Requests & API Integration:** Requests
- **Web Scraping:** BeautifulSoup4
- **Embedding Generation:** Sentence Transformers
- **Vector Database:** Pinecone
- **Generative API:** Gemini API (Google Generative Language API)
- **Web Search:** Google Custom Search API

## Setup

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
Install Dependencies:

Create and activate a virtual environment (optional but recommended):

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Then install the required packages:

bash
Copy
pip install -r requirements.txt
Configure Environment Variables:

Create a .env file in the root directory and add the following keys:

env
Copy
GOOGLE_API_KEY=<Your_Google_API_Key>
GEMINI_API_KEY=<Your_Gemini_API_Key>
PINECONE_API_KEY=<Your_Pinecone_API_Key>
CUSTOM_SEARCH_ENGINE_ID=<Your_Custom_Search_Engine_ID>
PINECONE_ENVIRONMENT=<Your_Pinecone_Environment>  # Optional if needed
Usage
1. Knowledge Base Construction
Run the knowledge base builder script to search for web content, extract educational insights, generate embeddings, and upsert the processed data into the Pinecone index:

bash
Copy
python knowledge_base_builder.py
When prompted, enter a topic. The script will perform a Google search, process the results, and store structured data in Pinecone.

2. Roadmap Generation
Run the roadmap generator script to create a day-by-day actionable learning plan based on the data stored in Pinecone:

bash
Copy
python roadmap_generator.py
When prompted, input the topic for which you want a roadmap. The script retrieves relevant context from the Pinecone index and then synthesizes the context into a detailed educational roadmap using the Gemini API.

Notes
Make sure the Pinecone index is named "knowledge-base-index" or update the scripts with your chosen index name.

The Gemini API expects a strict JSON output format; ensure that the prompt details in the scripts match the desired output structure.

Review API rate limits and adjust timeout settings if needed to suit your deployment or testing requirements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

yaml
Copy

---

Feel free to adjust sections such as installation steps, API configurations, and usage details to match your s
