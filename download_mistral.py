from huggingface_hub import snapshot_download
import os

print("Starting Mistral 7B download...")
print("This will take 10-20 minutes depending on your internet speed")
print("Model size: ~14-15 GB")

try:
    snapshot_download(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        local_dir='./models/mistral-7b',
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print("\n✅ Download completed successfully!")
    print("Model saved to: ./models/mistral-7b")
    
except Exception as e:
    print(f"\n❌ Error during download: {e}")
    print("Try running the script again - it will resume from where it stopped")