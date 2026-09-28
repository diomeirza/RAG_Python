from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# 1. Define a regular Python function. 
# The type hints and comments are critical—Gemini reads them like documentation!
def get_user_status(user_id: int) -> str:
    """
    Fetches the account status for a given user ID from the database.
    
    Args:
        user_id: The unique integer ID of the user.
    """
    print(f"\n[SYSTEM]: Executing get_user_status() for ID {user_id}...")
    
    # Mock database lookup
    mock_db = {
        101: "Active - Premium Member",
        202: "Suspended - Overdue Balance",
        303: "Informed - Pending Deletion"
    }
    
    return mock_db.get(user_id, "User not found")

# 2. Ask Gemini a question that requires your function
user_prompt = "Hey, can you check what's going on with user 202? Are they allowed to log in?"

print(f"User Asked: '{user_prompt}'")

# 3. Use our favorite generate_content method!
# We pass our function directly into the 'tools' configuration array.
response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents=user_prompt,
    config={
        'tools': [get_user_status] # Give Gemini access to your function
    }
)

# 4. Check if Gemini decided it needs to call your function
if response.function_calls:
    for call in response.function_calls:
        # Gemini tells you WHICH function to call and WHAT arguments to use
        if call.name == "get_user_status":
            # Extract arguments securely
            args = call.args
            
            # Execute your actual Python function using the AI's arguments
            result = get_user_status(user_id=int(args['user_id']))
            
            print(f"[SYSTEM]: Function returned: '{result}'")
            
            # 5. Send the result back to Gemini so it can formulate a natural final answer
            final_response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    user_prompt,                           # Original question
                    response.candidates[0].content,       # Gemini's function call intent
                    f"Function get_user_status returned: {result}" # The actual data
                ]
            )
            print(f"\nGemini's Final Answer:\n{final_response.text}")
else:
    # If the user just said "Hello", Gemini won't call a function and will just reply normally
    print(f"\nGemini's Answer:\n{response.text}")
