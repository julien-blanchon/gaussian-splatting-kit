from io import IOBase
import os
import subprocess
from typing import Optional
from pathlib import Path
from rich.console import Console

console = Console()

class FailedProcess(Exception):
    pass

def ffmpeg_extract_frames(
        video_path: Path,
        frames_path: Path,
        # TODO: Enable these options
        # start_time: Optional[str] = None,
        # duration: Optional[float] = None,
        # end_time: Optional[str]  = None,
        fps: float = 1,
        qscale: int = 1,
        stream_file: Optional[IOBase] = None
        ) -> str:
    frame_destination = frames_path / "input"
    console.log(f"🎞️  Extracting Images from {video_path} to {frame_destination} (fps: {fps}, qscale: {qscale}")
    # Create the directory to store the frames
    frames_path.mkdir(parents=True, exist_ok=True)
    frame_destination.mkdir(parents=True, exist_ok=True)
    # Store the current working directory
    cwd = os.getcwd()
    # Change the current working directory to frame_destination
    os.chdir(frame_destination)
    
    # Construct the ffmpeg command as a list of strings
    cmd = [
        'ffmpeg', 
        '-i', str(video_path), 
        '-qscale:v', str(qscale),
        '-qmin', '1',
        '-vf', f"fps={fps}",
        '%04d.jpg'
    ]

    console.log(f"💻 Executing command: {' '.join(cmd)}")
    
    _stdout = stream_file if stream_file else subprocess.PIPE
    with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
        if process.stdout:
            for line in process.stdout:
                print(line)
                
    # Change the current working directory back to the original
    os.chdir(cwd)

    return_code = process.returncode
    
    if return_code == 0:
        console.log(f"✅ Images Successfully Extracted! Path: {frames_path}")
    else:
        raise FailedProcess("Error extracting frames.")

    return frames_path

def ffmpeg_run(
        video_path: Path,
        output_path: Path,
        ffmpeg_command: str = "ffmpeg",
        # TODO: Enable these options
        # start_time: Optional[str] = None,
        # duration: Optional[float] = None,
        # end_time: Optional[str]  = None,
        fps: float = 1,
        qscale: int = 1,
        stream_file: Optional[IOBase] = None
        ) -> str:
    console.log("🌟 Starting the Frames Extraction...")
    frames_path = ffmpeg_extract_frames(
        video_path, 
        output_path,
        fps=fps, qscale=qscale, 
        stream_file=stream_file
    )
    console.log(f"🎉 Frames Extraction Complete! Path: {frames_path}")
    return frames_path

if __name__ == "__main__":
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+t') as temp_file:
        print(f"Using temp file: {temp_file.name}")
        try:
            ffmpeg_run(
                Path("/home/europe/Desktop/gaussian-splatting-kit/test/test.mov"),
                Path("/home/europe/Desktop/gaussian-splatting-kit/test"),
                stream_file=temp_file
            )
        except FailedProcess:
            console.log("🚨 Error extracting frames.")
            temp_file.seek(0)
            print(temp_file.read())

