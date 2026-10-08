import os
from google import genai
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()

# Initialize the client (it automatically picks up GEMINI_API_KEY from the environment)
client = genai.Client()

# Generate text
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents='Tell me a 1-sentence joke about programming.'
)

print(response.text)
