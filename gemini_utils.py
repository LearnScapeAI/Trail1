
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
