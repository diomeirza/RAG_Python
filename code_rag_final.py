import os
import glob
import time
from google import genai
from dotenv import load_dotenv
import chromadb
# Local, unlimited offline embedding transformer model
from sentence_transformers import SentenceTransformer

# 1. Initialization
load_dotenv(".env")
client = genai.Client()

# List out all target project directories for unified indexing
SOURCE_DIRECTORIES = [
    r"C:\BRI\Repos\brisurf\brisurf_api_controller",
    r"C:\BRI\Repos\brisurf\brisurf_api_model",      # Your Model project folder
    r"C:\BRI\Repos\brisurf\database_schema"         # Your exported SQL files folder
]
DB_STORAGE_PATH = "./chroma_db"

# Initialize local persistent storage
chroma_client = chromadb.PersistentClient(path=DB_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="brisurf_api_codebase")

# Load local embedding transformer (runs 100% on your machine CPU/RAM with zero rate limits)
print("📥 Loading local embedding model (all-MiniLM-L6-v2)...")
local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Unified, Delta-Tracking Multi-Directory Ingestion Block with Modification Detection
print("⚡ Scanning source directories for updates and changes...")

all_source_files = []

# Collect matching C# (.cs) and SQL (.sql) files from all paths
for directory in SOURCE_DIRECTORIES:
    if not os.path.exists(directory):
        print(f"⚠️ Warning: Configuration path not found, skipping: {directory}")
        continue
        
    # Gather C# files
    cs_pattern = os.path.join(directory, "**", "*.cs")
    all_source_files.extend(glob.glob(cs_pattern, recursive=True))
    
    # Gather SQL files
    sql_pattern = os.path.join(directory, "**", "*.sql")
    all_source_files.extend(glob.glob(sql_pattern, recursive=True))

total_files = len(all_source_files)

# Fetch previously indexed metadata records to build a tracking dictionary
# Structure: { file_path: last_modified_timestamp }
existing_meta = collection.get(include=["metadatas"])
db_file_tracker = {}

if existing_meta and existing_meta["metadatas"]:
    for meta in existing_meta["metadatas"]:
        if "file_path" in meta:
            # We track the maximum timestamp found for chunks of the same file
            path = meta["file_path"]
            mtime = meta.get("mtime", 0.0)
            if path not in db_file_tracker or mtime > db_file_tracker[path]:
                db_file_tracker[path] = mtime

files_to_process = []
modified_files_count = 0
new_files_count = 0

# Check each file on the drive against what is stored in the database
for file_path in all_source_files:
    try:
        current_mtime = os.path.getmtime(file_path)
    except Exception:
        continue  # Skip if file is locked or unreadable
        
    if file_path not in db_file_tracker:
        # Brand new file
        files_to_process.append((file_path, current_mtime, "New"))
        new_files_count += 1
    elif current_mtime > db_file_tracker[file_path]:
        # File has been modified since last index
        files_to_process.append((file_path, current_mtime, "Modified"))
        modified_files_count += 1

if files_to_process:
    print(f"📋 Scan results: Found {new_files_count} new files and {modified_files_count} modified files out of {total_files} total components.")
    print("Starting streaming local ingestion...\n")
    
    id_counter = collection.count()
    CHUNK_SIZE = 1500     
    CHUNK_OVERLAP = 200   
    
    for idx, (file_path, file_mtime, status) in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        
        # Categorize component type for clean logging output
        if file_path.endswith('.sql'):
            file_type = "SQL"
        elif "brisurf_api_model" in file_path.lower():
            file_type = "Model C#"
        else:
            file_type = "Controller C#"
            
        print(f"⚙️ [{idx + 1}/{len(files_to_process)}] [{status}] Indexing {file_type}: {filename}...", end="", flush=True)
        
        # CRITICAL STEP: If the file was modified, delete its OLD chunks first to avoid duplicates
        if status == "Modified":
            collection.delete(where={"file_path": file_path})
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read().strip()
            
            if not code_content:
                print(" (Skipped: Empty)")
                continue
            
            # Divide code blocks into overlapping context sliding windows
            file_chunks = []
            start = 0
            while start < len(code_content):
                end = start + CHUNK_SIZE
                file_chunks.append(code_content[start:end])
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            # Generate numerical vectors instantly using your computer
            chunk_vectors = local_embed_model.encode(file_chunks).tolist()
            
            # Save files permanently inside local database catalog
            for chunk_idx, (code_chunk, vector) in enumerate(zip(file_chunks, chunk_vectors)):
                ext_tag = "sql" if file_path.endswith('.sql') else "cs"
                
                collection.add(
                    ids=[f"chunk_{id_counter}"],
                    embeddings=[vector],
                    documents=[code_chunk],
                    metadatas={
                        "file_path": file_path, 
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "file_ext": ext_tag,
                        "mtime": file_mtime # Save timestamp to DB for future checks
                    }
                )
                id_counter += 1
            
            print(f" Done! -> Created {len(file_chunks)} chunks.")
            
        except Exception as e:
            print(f"\n⚠️ Error processing file {filename}: {e}")
            
    print(f"\n✅ Sync complete! Database now holds {collection.count()} total unified chunks.")
else:
    print(f"📦 Database perfectly synced. All {total_files} architecture components are up-to-date.")

# 3. Interactive Code Explorer Chat Loop with Memory
print("\n" + "="*50)
print("🤖 Enterprise C# & SQL Explorer Active (With Memory)")
print("Type 'exit' or 'quit' to close the assistant.")
print("="*50)

system_instruction = """
You are an expert enterprise .NET backend architect and database administrator. 
Analyze the provided local C# code and SQL context snippets to explain multi-layer request flows, 
pinpoint controller-to-model validation mismatches, trace Stored Procedure implementations, 
and explain specific logic paths accurately. Rely ONLY on the provided snippets. Remember prior questions.
"""

# Initialize stateful conversation stream
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
        # Generate search query matching vector using local embedding transformer
        query_vector = local_embed_model.encode(user_query).tolist()
        
        # Query ChromaDB database for top 4 closest semantic matches across all layers
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=4
        )
        
        retrieved_documents = search_results["documents"]
        retrieved_metadata = search_results["metadatas"]
        
        context_str = ""
        # Handle cross-version database result array extraction styles securely
        docs = retrieved_documents if retrieved_documents and isinstance(retrieved_documents, list) else retrieved_documents
        metas = retrieved_metadata if retrieved_metadata and isinstance(retrieved_metadata, list) else retrieved_metadata
        
        # Format the context block string layout
        for doc, meta in zip(docs, metas):
            # Account for metadata sometimes being inside nested lists depending on result sizes
            current_meta = meta[0] if isinstance(meta, list) else meta
            context_str += f"\nFile Path: {current_meta['file_path']}\nCode Snippet:\n{doc}\n=====================\n"
            
        # Compile final multi-layer augmented prompt
        rag_prompt = f"""
Context from 'brisurf' workspace:
---------------------------------------------
{context_str}
---------------------------------------------

User Request: {user_query}
"""
        
        # Dispatch context message directly to the stateful tracking system session
        response = chat_session.send_message(rag_prompt)
        
        print("\n🤖 Assistant Response:")
        print(response.text)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
