import os
from dotenv import load_dotenv
# Import classes from your new modular files
from database import VectorDBManager
from embedder import LocalEmbedder
from ingestion import CodebaseScanner
from agent import GeminiAgent

def main():
    # Load configuration environments
    load_dotenv(".env")
    
    # Configure project paths
    SOURCE_DIRECTORIES = [
        r"C:\BRI\Repos\brisurf\brisurf_api_controller",
        r"C:\BRI\Repos\brisurf\brisurf_api_model"
    ]
    
    # 1. Initialize Components (Dependency Wiring)
    db_manager = VectorDBManager()
    embedder = LocalEmbedder()
    
    # 2. Run Indexing Sync Pipeline
    scanner = CodebaseScanner(SOURCE_DIRECTORIES, db_manager, embedder)
    scanner.scan_and_sync()
    
    # 3. Spin up AI reasoning conversational layer
    agent = GeminiAgent()
    
    print("\n" + "="*50)
    print("🤖 Enterprise C# Workspace Explorer Active (Modular Edition)")
    print("Type 'exit' or 'quit' to close the assistant.")
    print("="*50)

    # 4. Main Interactive Execution Loop
    while True:
        user_query = input("\n🔎 Ask something about your codebase: ").strip()
        
        if not user_query:
            continue
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        print("Thinking...")
        
        try:
            # Generate search parameters locally
            query_vector = embedder.generate_single_embedding(user_query)
            
            # Query Database Layer for matching elements
            search_results = db_manager.query_semantic_matches(query_vector, n_results=5)
            
            context_str = ""
            
            # --- FIX: ROBUSTLY UNWRAP CHROMADB'S DOUBLE-NESTED RETURN ARRAYS ---
            if search_results and "documents" in search_results and search_results["documents"]:
                # Extract the first element to get the actual list of documents/metadata strings cleanly
                inner_docs = search_results["documents"][0] if isinstance(search_results["documents"][0], list) else search_results["documents"]
                inner_metas = search_results["metadatas"][0] if isinstance(search_results["metadatas"][0], list) else search_results["metadatas"]
                
                # Format extracted text pieces into unified context blocks
                for doc, meta in zip(inner_docs, inner_metas):
                    if meta and "file_path" in meta:
                        context_str += f"\nFile Path: {meta['file_path']}\nCode Snippet:\n{doc}\n=====================\n"
            
            if not context_str:
                context_str = "No specific code chunks were retrieved matching this semantic vector query."

            # Fetch result from the reasoning engine agent
            assistant_response = agent.ask(user_query, context_str)
            
            print("\n🤖 Assistant Response:")
            print(assistant_response)
            print("-" * 40)

            
        except Exception as e:
            print(f"❌ An error occurred during runtime processing: {e}")

if __name__ == "__main__":
    main()
