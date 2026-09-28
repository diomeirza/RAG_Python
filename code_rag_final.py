import os
import glob
import time
from google import genai
from dotenv import load_dotenv
import chromadb
# 1. Import the local transformer engine
from sentence_transformers import SentenceTransformer

# Initialization
load_dotenv(".env")
client = genai.Client()

CODEBASE_PATH = r"C:\BRI\Repos\brisurf\brisurf_api_controller"
DB_STORAGE_PATH = "./chroma_db"

# Initialize local persistent ChromaDB storage
chroma_client = chromadb.PersistentClient(path=DB_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="brisurf_api_codebase")

# 2. Load the Local Embedding Model (Runs 100% offline on your CPU/RAM)
print("📥 Loading local embedding model (all-MiniLM-L6-v2)...")
local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

existing_count = collection.count()

# Incremental Local Ingestion Block
print(f"⚡ Scanning {CODEBASE_PATH} for new files...")
search_pattern = os.path.join(CODEBASE_PATH, "**", "*.cs")
csharp_files = glob.glob(search_pattern, recursive=True)
total_files = len(csharp_files)

# Fetch previously indexed files to allow delta-syncing
existing_meta = collection.get(include=["metadatas"])
indexed_paths = set(meta["file_path"] for meta in existing_meta["metadatas"] if "file_path" in meta) if existing_meta else set()
new_files = [f for f in csharp_files if f not in indexed_paths]

if new_files:
    print(f"📋 Found {len(new_files)} unindexed C# files. Processing locally with ZERO rate limits...\n")
    
    id_counter = collection.count()
    CHUNK_SIZE = 1500     
    CHUNK_OVERLAP = 200   
    
    for idx, file_path in enumerate(new_files):
        filename = os.path.basename(file_path)
        print(f"⚙️ [{idx + 1}/{len(new_files)}] Indexing: {filename}...", end="", flush=True)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read().strip()
            
            if not code_content:
                print(" (Skipped: Empty)")
                continue
            
            # Chunk long C# files into sliding windows
            file_chunks = []
            start = 0
            while start < len(code_content):
                end = start + CHUNK_SIZE
                file_chunks.append(code_content[start:end])
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            # Generate vectors offline using your local CPU
            chunk_vectors = local_embed_model.encode(file_chunks).tolist()
            
            # Save chunks to ChromaDB
            for chunk_idx, (code_chunk, vector) in enumerate(zip(file_chunks, chunk_vectors)):
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
            
            print(f" Done! -> Created {len(file_chunks)} chunks.")
            
        except Exception as e:
            print(f"\n⚠️ Error on file {filename}: {e}")
            
    print(f"\n✅ Sync complete! Database has {collection.count()} total chunks.")
else:
    print(f"📦 Database up to date. All {total_files} files are already indexed.")

# 3. Interactive Runtime Query Loop with Memory
print("\n" + "="*50)
print("🤖 C# Code Explorer Terminal Active (With Context Memory)")
print("Type 'exit' or 'quit' to close the assistant.")
print("="*50)

system_instruction = """
You are an expert enterprise .NET backend architect and developer. 
Analyze the provided local C# code contexts to explain architecture flows, pinpoint error sources, 
and summarize specific business logic paths accurately. Remember what the user asked previously.
"""

# Initialize a stateful multi-turn chat session with Gemini
chat_session = client.chats.create(
    model="gemini-3.6-flash",
    config={"system_instruction": system_instruction}
)

while True:
    user_query = input("\n🔎 Ask something about your codebase: ").strip()
    
    if not user_query:
        continue
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    print("Thinking...")
    
    try:
        # --- FIXED STEP: Convert runtime query using the SAME local model ---
        # Instead of calling Google's API, we generate the vector instantly on your machine
        query_vector = local_embed_model.encode(user_query).tolist()
        
        # Query ChromaDB using your local query vector
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        # Flatten and aggregate the relevant files found
        retrieved_documents = search_results["documents"]
        retrieved_metadata = search_results["metadatas"]
        
        context_str = ""
        # Handle cases where search results are nested inside arrays
        docs = retrieved_documents[0] if retrieved_documents and isinstance(retrieved_documents[0], list) else retrieved_documents
        metas = retrieved_metadata[0] if retrieved_metadata and isinstance(retrieved_metadata[0], list) else retrieved_metadata
        
        for doc, meta in zip(docs, metas):
            context_str += f"\nFile Path: {meta['file_path']}\nCode Snippet:\n{doc}\n=====================\n"
            
        rag_prompt = f"""
Context from 'brisurf_api_controller' repository:
---------------------------------------------
{context_str}
---------------------------------------------

User Request: {user_query}
"""
        
        # Send the contextually loaded prompt to Gemini's reasoning layer
        response = chat_session.send_message(rag_prompt)
        
        print("\n🤖 Assistant Response:")
        print(response.text)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
