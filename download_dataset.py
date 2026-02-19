import argparse
from huggingface_hub import snapshot_download
import os

def download_folder(repo_id, folder_path, local_dir, repo_type="dataset"):
    """
    Downloads a specific folder from a Hugging Face repository.
    
    Args:
        repo_id (str): The repository ID (e.g., 'username/dataset_name').
        folder_path (str): The folder path in the repo to download.
        local_dir (str): The local directory to save to.
        repo_type (str): Type of the repo ('dataset', 'model', or 'space').
    """
    print(f"Starting download...")
    print(f"Repo: {repo_id}")
    print(f"Folder: {folder_path}")
    print(f"Local Dir: {local_dir}")
    
    # Ensure the pattern matches files inside the folder recursively
    # if folder_path is "foo", we want "foo/**"
    if not folder_path.endswith("*"):
        if folder_path.endswith("/"):
             allow_patterns = f"{folder_path}**"
        else:
             allow_patterns = f"{folder_path}/**"
    else:
        allow_patterns = folder_path

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=None # Uses cached token if logged in, or None for public
        )
        print(f"\nSuccessfully downloaded '{folder_path}' from {repo_id} to {local_dir}")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a specific folder from a Hugging Face repository.")
    
    parser.add_argument("repo_id", type=str, help="The repository ID (e.g., 'OpenSLR/librispeech_asr')")
    parser.add_argument("folder", type=str, help="The specific folder path to download (e.g., 'data/train-clean-100')")
    parser.add_argument("--local_dir", type=str, default="./downloads", help="Local directory to save files (default: ./downloads)")
    parser.add_argument("--type", type=str, default="dataset", choices=["dataset", "model", "space"], help="Repository type (default: dataset)")

    args = parser.parse_args()
    
    download_folder(args.repo_id, args.folder, args.local_dir, args.type)
