from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# 1. Define your C#-like "Model" or "POCO"
class TicketAnalysis(BaseModel):
    sentiment: str = Field(description="Must be either Positive, Neutral, or Negative")
    urgency_score: int = Field(description="Scale of 1 to 5, where 5 is extremely critical")
    summary: str = Field(description="A clean 1-sentence summary of the customer's problem")
    key_tags: list[str] = Field(description="List of keywords like 'billing', 'bug', 'ui'")

# 2. Raw input data that an application might receive
customer_email = """
Hey, I've been trying to log in for the past 2 hours and your system keeps throwing a 500 error. 
I am losing money because my clients can't access their dashboards. Please fix this ASAP!
"""

print("Analyzing ticket...")

# 3. Call Gemini and FORCE it to match your schema
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents=f"Analyze this customer email: {customer_email}",
    config={
        'response_mime_type': 'application/json',
        'response_schema': TicketAnalysis,
    }
)

# 4. Access the properties safely using standard object dot-notation
# Python handles the deserialization automatically via the SDK!
analysis: TicketAnalysis = response.parsed

print("\n--- Results ---")
print(f"Sentiment: {analysis.sentiment}")
print(f"Urgency:   {analysis.urgency_score}/5")
print(f"Summary:   {analysis.summary}")
print(f"Tags:      {', '.join(analysis.key_tags)}")
