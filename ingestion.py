import os
import glob
import time

class CodebaseScanner:
    def __init__(self, source_directories, db_manager, embedder, chunk_size=1500, chunk_overlap=200):
        self.source_directories = source_directories
        self.db_manager = db_manager
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_file_tracker_map(self):
        """Builds an in-memory map of { normalized_path: integer_timestamp } from DB."""
        db_file_tracker = {}
        all_metadata = self.db_manager.get_all_metadata()
        
        for meta in all_metadata:
            if meta and "file_path" in meta:
                path = os.path.normpath(meta["file_path"])
                mtime = int(meta.get("mtime", 0))
                if path not in db_file_tracker or mtime > db_file_tracker[path]:
                    db_file_tracker[path] = mtime
        return db_file_tracker

    def scan_and_sync(self):
        """Scans workspace projects, filters changed entries, and updates indexes."""
        print("⚡ Scanning C# directories for updates and changes...")
        all_source_files = []
        
        for directory in self.source_directories:
            if not os.path.exists(directory):
                print(f"⚠️ Warning: Path not found, skipping: {directory}")
                continue
            cs_pattern = os.path.join(directory, "**", "*.cs")
            all_source_files.extend(glob.glob(cs_pattern, recursive=True))

        total_files = len(all_source_files)
        db_file_tracker = self.build_file_tracker_map()

        files_to_process = []
        new_count = 0
        modified_count = 0

        for file_path in all_source_files:
            native_path = os.path.normpath(file_path)
            try:
                current_mtime = int(os.path.getmtime(native_path))
            except Exception:
                continue

            if native_path not in db_file_tracker:
                files_to_process.append((native_path, current_mtime, "New"))
                new_count += 1
            elif current_mtime > db_file_tracker[native_path]:
                files_to_process.append((native_path, current_mtime, "Modified"))
                modified_count += 1

        if not files_to_process:
            print(f"📦 Database perfectly synced. All {total_files} components match current drive states.")
            return

        print(f"📋 Scan results: Found {new_count} new files and {modified_count} modified files out of {total_files} total components.")
        print("Starting streaming local ingestion pipeline...\n")

        id_counter = int(time.time() * 1000)

        for idx, (file_path, file_mtime, status) in enumerate(files_to_process):
            if "brisurf_api_model" in file_path.lower() or "model" in file_path.lower():
                file_type = "Model C#"
            else:
                file_type = "Controller C#"

            print(f"⚙️ [{idx + 1}/{len(files_to_process)}] [{status}] Indexing {file_type}: {os.path.basename(file_path)}...", end="", flush=True)

            if status == "Modified":
                self.db_manager.delete_file_records(file_path)

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_content = f.read().strip()

                if not code_content:
                    print(" (Skipped: Empty)")
                    continue

                # Sliding Context Window Chunker
                file_chunks = []
                start = 0
                while start < len(code_content):
                    end = start + self.chunk_size
                    file_chunks.append(code_content[start:end])
                    start += (self.chunk_size - self.chunk_overlap)

                if file_chunks:
                    chunk_vectors = self.embedder.generate_embeddings(file_chunks)
                    
                    chunk_ids = []
                    metadatas = []
                    for chunk_idx, code_chunk in enumerate(file_chunks):
                        chunk_ids.append(f"chunk_{id_counter}")
                        metadatas.append({
                            "file_path": file_path,
                            "filename": os.path.basename(file_path),
                            "chunk_index": chunk_idx,
                            "file_ext": "cs",
                            "mtime": file_mtime
                        })
                        id_counter += 1

                    self.db_manager.add_segments(chunk_ids, chunk_vectors, file_chunks, metadatas)
                    print(f" Done! -> Created {len(file_chunks)} segment(s).")
                else:
                    print(" (Skipped: No valid segments)")

            except Exception as e:
                print(f"\n⚠️ Error processing file {os.path.basename(file_path)}: {e}")

        print(f"\n✅ Sync complete! Database now holds {self.db_manager.get_total_chunks()} total unified chunks.")
