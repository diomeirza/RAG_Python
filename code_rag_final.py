import os
import glob
import time
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
import chromadb

# 1. Initialization
load_dotenv(".env")
client = genai.Client()

CODEBASE_PATH = r"C:\BRI\Repos\brisurf\brisurf_api_controller\App_Code"
DB_STORAGE_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=DB_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="brisurf_api_codebase")

existing_count = collection.count()

# 2. Safe, Rate-Limited Indexing Sequence
# 2. Incremental, Safe Ingestion Block
print(f"⚡ Scanning {CODEBASE_PATH} for new files...")

search_pattern = os.path.join(CODEBASE_PATH, "**", "*.cs")
csharp_files = glob.glob(search_pattern, recursive=True)
total_files = len(csharp_files)

# Fetch all existing unique file paths currently stored in your database metadata
# This allows us to do a lightning-fast O(1) lookup in Python memory
existing_meta = collection.get(include=["metadatas"])
indexed_paths = set()
if existing_meta and existing_meta["metadatas"]:
    for meta in existing_meta["metadatas"]:
        if "file_path" in meta:
            indexed_paths.add(meta["file_path"])

# Filter down to files that aren't in the database yet
new_files = [f for f in csharp_files if f not in indexed_paths]

if new_files:
    print(f"📋 Found {len(new_files)} new/unindexed C# files out of {total_files} total files.")
    print("Starting streaming ingestion for new files...\n")
    
    # Start our chunk tracking ID based on how many records already exist
    id_counter = collection.count()
    CHUNK_SIZE = 1500     
    CHUNK_OVERLAP = 200   
    
    for idx, file_path in enumerate(new_files):
        filename = os.path.basename(file_path)
        print(f"⚙️ [{idx + 1}/{len(new_files)}] Processing New File: {filename}...", end="", flush=True)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read().strip()
            
            if not code_content:
                print(" (Skipped: Empty file)")
                continue
            
            # Chunking text
            file_chunks = []
            start = 0
            while start < len(code_content):
                end = start + CHUNK_SIZE
                file_chunks.append(code_content[start:end])
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            # --- UPDATE THE INNER EMBEDDING LOOP TO MATCH THIS RESILIENT VERSION ---

            # Vectorize each chunk
            for chunk_idx, code_chunk in enumerate(file_chunks):
                while True:
                    try:
                        embedding_response = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=code_chunk
                        )
                        break # Success! Break the retry loop
                        
                    except APIError as e:
                        if e.code == 429:
                            print("\n⏳ Rate limit hit. Pausing 60 seconds to reset quota...")
                            time.sleep(60)
                        else:
                            raise e
                            
                    except Exception as net_error:
                        # Catches "getaddrinfo failed", DNS timeouts, or Wi-Fi dropouts
                        if "11001" in str(net_error) or "getaddrinfo" in str(net_error).lower():
                            print(f"\n🌐 Network connection dropped ([Errno 11001]). Retrying in 15 seconds...")
                            time.sleep(15)
                        else:
                            # If it's a completely different unrecoverable error, raise it
                            raise net_error
                
                if hasattr(embedding_response, 'embeddings') and hasattr(embedding_response.embeddings, 'values'):
                    vector = embedding_response.embeddings.values
                elif isinstance(embedding_response.embeddings, list) and hasattr(embedding_response.embeddings, 'values'):
                    vector = embedding_response.embeddings.values
                elif isinstance(embedding_response.embeddings, list):
                    vector = embedding_response.embeddings
                else:
                    vector = embedding_response.embeddings.values if hasattr(embedding_response.embeddings, 'values') else embedding_response.embeddings

                collection.add(
                    ids=[f"chunk_{id_counter}"],
                    embeddings=[vector],
                    documents=[code_chunk],
                    metadatas={
                        "file_path": file_path, 
                        "filename": filename,
                        "chunk_index": chunk_idx
                    }
                )
                id_counter += 1
            
            print(f" Done! -> Generated {len(file_chunks)} chunks.")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n⚠️ Error on file {filename}: {e}")
            
    print(f"\n✅ Sync complete! Database now has {collection.count()} total indexed chunks.")
else:
    print(f"📦 Database up to date. All {total_files} files are already indexed. Skipping ingestion.")


# 3. Interactive Runtime Query Loop
print("\n" + "="*50)
print("🤖 C# Code Explorer Terminal Active")
print("Type 'exit' or 'quit' to close the assistant.")
print("="*50)

system_instruction = """
You are an expert enterprise .NET backend architect and developer. 
Analyze the provided local C# code contexts to explain architecture flows, pinpoint error sources, 
and summarize specific business logic paths accurately.
"""

while True:
    user_query = input("\n🔎 Ask something about your codebase: ").strip()
    
    if not user_query:
        continue
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    print("Thinking...")
    
    try:
        # Convert user runtime query to vector space
        query_embedding_response = client.models.embed_content(
            model="text-embedding-004",
            contents=user_query
        )
        if hasattr(query_embedding_response.embeddings, 'values'):
            query_vector = query_embedding_response.embeddings.values
        elif isinstance(query_embedding_response.embeddings, list):
            query_vector = query_embedding_response.embeddings[0].values if hasattr(query_embedding_response.embeddings[0], 'values') else query_embedding_response.embeddings[0]
        else:
            query_vector = query_embedding_response.embeddings
        
        # Pull top 3 relevant file classes for richer context mapping
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        # Flatten findings
        retrieved_documents = search_results["documents"][0]
        retrieved_metadata = search_results["metadatas"][0]
        
        context_str = ""
        for doc, meta in zip(retrieved_documents, retrieved_metadata):
            context_str += f"\nFile Path: {meta['file_path']}\nCode Snippet:\n{doc}\n=====================\n"
            
        rag_prompt = f"""
Context from 'brisurf_api_controller' repository:
---------------------------------------------
{context_str}
---------------------------------------------

Question: {user_query}
"""
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=rag_prompt,
            config={"system_instruction": system_instruction}
        )
        
        print("\n🤖 Assistant Response:")
        print(response.text)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
