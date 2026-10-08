from google import genai

class GeminiAgent:
    def __init__(self, model_name="gemini-3.6-flash"):
        self.model_name = model_name
        self.client = genai.Client()
        
        self.system_instruction = """
        You are an expert enterprise .NET backend architect and developer. 
        Analyze the provided local C# code snippets to explain request flows, pinpoint controller-to-model validation 
        mismatches, trace data transformation bindings, and accurately break down business logic rules. 
        Remember prior questions.
        """
        
        # Initialize stateful conversation stream wrapper 
        self.chat_session = self.client.chats.create(
            model=self.model_name,
            config={"system_instruction": self.system_instruction}
        )

    def ask(self, user_query, context_str):
        """Assembles prompt and returns state-tracked chat token string streams."""
        rag_prompt = f"""
        Context from 'brisurf' workspace:
        ---------------------------------------------
        {context_str}
        ---------------------------------------------

        User Request: {user_query}
        """
        response = self.chat_session.send_message(rag_prompt)
        return response.text
