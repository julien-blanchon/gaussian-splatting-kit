from pathlib import Path
import requests
from rich.console import Console

console = Console()

def download_file(url: str, file_path: Path) -> Path:
    console.log(f"📥 Downloading File from URL: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with file_path.open('wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        console.log(f"✅ File Successfully Downloaded! Path: {file_path}")
    else:
        console.log(f"🚨 Error downloading file from {url}.")
    return file_path

def download_api(url: str, file_path: Path) -> Path:
    # Download the video from internet
    video_path = file_path + '/video.mp4'
    console.log("🌟 Starting the Video Download...")
    video_path = download_file(url, video_path)
    console.log(f"🎉 Video Download Complete! Path: {video_path}")
    return video_path