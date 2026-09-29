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
    r"C:\BRI\Repos\brisurf\brisurf_api_model",      
    r"C:\BRI\Repos\brisurf\database_schema"         
]
DB_STORAGE_PATH = "./chroma_db"

# Initialize local persistent storage
chroma_client = chromadb.PersistentClient(path=DB_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="brisurf_api_codebase")

# Load local embedding transformer
print("📥 Loading local embedding model (all-MiniLM-L6-v2)...")
local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Unified, Delta-Tracking Multi-Directory Ingestion Block
print("⚡ Scanning source directories for updates and changes...")

all_source_files = []

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

# Fetch total record count to lift Chroma's default limit restriction
total_chunks_in_db = collection.count()
db_file_tracker = {}

if total_chunks_in_db > 0:
    # Lift the 100-row limit barrier to scan every single tracked file fragment
    existing_meta = collection.get(include=["metadatas"], limit=total_chunks_in_db)
    
    if existing_meta and "metadatas" in existing_meta and existing_meta["metadatas"]:
        outer_list = existing_meta["metadatas"]
        inner_list = outer_list if isinstance(outer_list, list) else outer_list
        
        for meta in inner_list:
            if meta and "file_path" in meta:
                # FIX 1: Enforce clean cross-platform forward slashes for DB strings
                path = meta["file_path"].replace("\\", "/")
                # FIX 2: Enforce strong integer data casting to kill float precision bugs
                mtime = int(meta.get("mtime", 0))
                # Keep the absolute highest timestamp found across matching chunks
                if path not in db_file_tracker or mtime > db_file_tracker[path]:
                    db_file_tracker[path] = mtime

files_to_process = []
modified_files_count = 0
new_files_count = 0

for file_path in all_source_files:
    # FIX 3: Enforce identical cross-platform forward slashes on active files
    clean_path = file_path.replace("\\", "/")
    try:
        # FIX 4: Convert live disk timestamps to absolute integers
        current_mtime = int(os.path.getmtime(clean_path))
    except Exception:
        continue  
        
    if clean_path not in db_file_tracker:
        files_to_process.append((clean_path, current_mtime, "New"))
        new_files_count += 1
    elif current_mtime > db_file_tracker[clean_path]:
        files_to_process.append((clean_path, current_mtime, "Modified"))
        modified_files_count += 1

if files_to_process:
    print(f"📋 Scan results: Found {new_files_count} new files and {modified_files_count} modified files out of {total_files} total components.")
    print("Starting streaming local ingestion...\n")
    
    # Generate unique absolute IDs to prevent ID overlap collapses when chunks are deleted
    id_counter = int(time.time() * 1000)
    CHUNK_SIZE = 1500     
    CHUNK_OVERLAP = 200   
    
    for idx, (file_path, file_mtime, status) in enumerate(files_to_process):
        filename = os.path.basename(file_path).lower()
        
        if filename.endswith('table.sql'):
            file_type = "SQL Table"
        elif filename.endswith('storeprocedure.sql'):
            file_type = "SQL SP"
        elif "brisurf_api_model" in file_path.lower() or "model" in file_path.lower():
            file_type = "Model C#"
        else:
            file_type = "Controller C#"
            
        print(f"⚙️ [{idx + 1}/{len(files_to_process)}] [{status}] Indexing {file_type}: {os.path.basename(file_path)}...", end="", flush=True)
        
        # Safe delta cleanup using clean forward slash lookup standards
        if status == "Modified":
            collection.delete(where={"file_path": file_path})
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read().strip()
            
            if not code_content:
                print(" (Skipped: Empty)")
                continue
            
            file_chunks = []
            
            # Keep table script fields fully intact
            if file_type == "SQL Table":
                file_chunks.append(code_content)
            else:
                # Sliding context slice mechanism
                start = 0
                while start < len(code_content):
                    end = start + CHUNK_SIZE
                    file_chunks.append(code_content[start:end])
                    start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            # Vectorize items locally completely free
            if file_chunks:
                chunk_vectors = local_embed_model.encode(file_chunks).tolist()
                
                for chunk_idx, (code_chunk, vector) in enumerate(zip(file_chunks, chunk_vectors)):
                    ext_tag = "sql" if file_path.endswith('.sql') else "cs"
                    
                    collection.add(
                        ids=[f"chunk_{id_counter}"],
                        embeddings=[vector],
                        documents=[code_chunk],
                        metadatas={
                            "file_path": file_path, # Clean forward slashes
                            "filename": os.path.basename(file_path),
                            "chunk_index": chunk_idx,
                            "file_ext": ext_tag,
                            "mtime": file_mtime  
                        }
                    )
                    id_counter += 1
                    
                print(f" Done! -> Created {len(file_chunks)} segment(s).")
            else:
                print(" (Skipped: No valid segments)")
            
        except Exception as e:
            print(f"\n⚠️ Error processing file {os.path.basename(file_path)}: {e}")
            
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
and accurately describe table structures (columns, constraints, keys). Rely ONLY on the provided snippets. 
Remember prior questions. If the user asks about a table structure, pull it directly from the SQL context snippets.
"""

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
        # Generate match vectors instantly offline
        query_vector = local_embed_model.encode(user_query).tolist()
        
        # Retrieve top 5 cross-project matches
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )
        
        context_str = ""
        
        if search_results and "documents" in search_results and search_results["documents"]:
            inner_docs = search_results["documents"]
            inner_metas = search_results["metadatas"]
            
            for doc, meta in zip(inner_docs, inner_metas):
                if meta and "file_path" in meta:
                    context_str += f"\nFile Path: {meta['file_path']}\nCode Snippet:\n{doc}\n=====================\n"
        
        if not context_str:
            context_str = "No specific code chunks were retrieved matching this semantic vector search query."

        rag_prompt = f"""
Context from 'brisurf' workspace:
---------------------------------------------
{context_str}
---------------------------------------------

User Request: {user_query}
"""
        
        response = chat_session.send_message(rag_prompt)
        
        print("\n🤖 Assistant Response:")
        print(response.text)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
