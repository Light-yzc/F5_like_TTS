from huggingface_hub import list_repo_files
import argparse

def list_files(repo_id, repo_type="dataset"):
    try:
        print(f"Listing files for {repo_id}...")
        files = list_repo_files(repo_id, repo_type=repo_type)
        print(f"Found {len(files)} files.")
        for f in files:
            if "Premium" in f:
                print(f)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_files("Wenetspeech4TTS/WenetSpeech4TTS")
