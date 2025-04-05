from google import genai

# Initialize the client with your API key
client = genai.Client(api_key="AIzaSyA8zJq4rCy_fY28QRBchRY6a3paSASTZeE")

# Make a request using the generate_content method
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="Explain how AI works in a few words"
)

print(response.text)
