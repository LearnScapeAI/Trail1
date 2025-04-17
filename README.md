
# 🧠 AI-Powered Knowledge Base & Learning Roadmap Generator

This project builds a personalized, topic-specific learning roadmap using:
- 🔍 Web search results
- 🤖 Gemini LLM API
- 📺 YouTube videos
- 🧠 Pinecone vector database

---

## 🚀 Features

- **Modular Architecture**: Core logic is split across `gemini_utils.py`, `pinecone_utils.py`, `youtube_utils.py`, and `main.py`.
- **Day-by-Day Learning Plans**: Generates structured JSON roadmaps with individual tasks per day.
- **YouTube Enrichment**: Every subtopic/task is enhanced with relevant video links (duration < 10 mins).
- **Vector Search with Pinecone**: Retrieves relevant summaries from previously stored knowledge.
- **Custom Parameters**:
  - Skill level (Beginner / Intermediate / Advanced)
  - Time commitment per day
  - Total duration (in days)
- **Structured Logging**: All runtime events use Python’s `logging` for traceability.
- **Environment Configurable**: Uses `.env` for secrets and keys.

---

## 🗂️ Project Structure

```bash
.
├── main.py                  # CLI entrypoint
├── gemini_utils.py          # Handles Gemini LLM calls
├── pinecone_utils.py        # Embedding + query functions for Pinecone
├── youtube_utils.py         # YouTube API integration for short videos
├── .env                     # API Keys (Google, Gemini, Pinecone)
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Setup

1. **Clone the repo** and set up a virtual environment:
   ```bash
   git clone <your-repo-url>
   cd <your-project>
   python -m venv venv && source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with:
   ```env
   GOOGLE_API_KEY=your_google_key
   GEMINI_API_KEY=your_gemini_key
   YOUTUBE_API_KEY=your_youtube_key
   PINECONE_API_KEY=your_pinecone_key
   ```

4. **Run the program**:
   ```bash
   python main.py
   ```

---

## 🛠 Example Usage

```bash
Enter topic: system design
Skill level (Beginner/Intermediate/Advanced): beginner
Daily time commitment (e.g., '2 hours'): 1.5
Roadmap duration (days): 30
```

✅ The output will be a 30-day JSON roadmap enriched with YouTube links.

---

## 📝 Logging

- Logging is configured in `main.py`:
  ```python
  logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
  ```

- You can change `level` to `DEBUG` to print raw LLM outputs.

---

## 🧪 Coming Soon

- `tests/` directory with `pytest` support
- File output & resume export
- PDF roadmap generation
- LangGraph node orchestration (multi-stage flow)

---

## 📌 Dependencies

- `python-dotenv`
- `requests`
- `isodate`
- `sentence-transformers`
- `pinecone-client`

Install via:
```bash
pip install -r requirements.txt
```

---
