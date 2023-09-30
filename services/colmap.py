from typing import Literal, Optional
from io import IOBase
import os
from pathlib import Path
import shutil
import subprocess
from rich.progress import Progress
from rich.console import Console

console = Console()

class FailedProcess(Exception):
    pass

def colmap_feature_extraction(
        database_path: Path, 
        image_path: Path, 
        camera: Literal["OPENCV"], 
        colmap_command: str = "colmap", 
        use_gpu: bool = True,
        stream_file: Optional[IOBase] = None
    ):
    total = len(list(image_path.glob("*.jpg")))
    with Progress(console=console) as progress:
        task = progress.add_task("Feature Extraction", total=total)

        database_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            colmap_command,
            "feature_extractor",
            "--database_path", database_path.as_posix(),
            "--image_path", image_path.as_posix(),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera,
            "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
            # "--SiftExtraction.domain_size_pooling", "1",
            # "--SiftExtraction.estimate_affine_shape", "1"
        ]
        console.log(f"💻 Executing command: {' '.join(cmd)}")
        
        _stdout = stream_file if stream_file else subprocess.PIPE
        with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    if line.startswith("Processed file "):
                        line_process = line\
                            .replace("Processed file [", "")\
                            .replace("]", "")\
                            .replace("\n", "")
                        current, total = line_process.split("/")
                        progress.update(task, completed=int(current), total=int(total), refresh=True)

        progress.update(task, completed=int(total), refresh=True)

    return_code = process.returncode
    
    if return_code == 0:
        
        console.log(f'Feature stored in {database_path.as_posix()}.')
        console.log('✅ Feature extraction completed.')
    else:
        raise FailedProcess("Feature extraction failed.")

def colmap_feature_matching(
        database_path: Path,
        image_path: Path,
        colmap_command: str = "colmap",
        use_gpu: bool = True,
        stream_file: Optional[IOBase] = None
    ):
    total = len(list(image_path.glob("*.jpg")))
    with Progress(console=console) as progress:
        task = progress.add_task("Feature Matching", total=total)

        database_path
        cmd = [
            colmap_command,
            "exhaustive_matcher",
            "--database_path", database_path.as_posix(),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0"
        ]
        console.log(f"💻 Executing command: {' '.join(cmd)}")
        
        _stdout = stream_file if stream_file else subprocess.PIPE
        with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    pass

        progress.update(task, completed=int(total), refresh=True)
    
    return_code = process.returncode

    if return_code == 0:
        
        console.log('✅ Feature matching completed.')
    else:
        raise FailedProcess("Feature matching failed.")

def colmap_bundle_adjustment(
        database_path: Path,
        image_path: Path,
        sparse_path: Path,
        colmap_command: str = "colmap",
        stream_file: Optional[IOBase] = None
    ):
    total = len(list(image_path.glob("*.jpg")))
    with Progress(console=console) as progress:
        task = progress.add_task("Bundle Adjustment", total=total)

        cmd = [
            colmap_command,
            "mapper",
            "--database_path", database_path.as_posix(),
            "--image_path", image_path.as_posix(),
            "--output_path", sparse_path.as_posix(),
            "--Mapper.ba_global_function_tolerance=0.000001"
            # "--Mapper.ba_local_max_num_iterations", "40",
            # "--Mapper.ba_global_max_num_iterations", "100",
            # "--Mapper.ba_local_max_refinements", "3",
            # "--Mapper.ba_global_max_refinements", "5"
        ]
        console.log(f"💻 Executing command: {' '.join(cmd)}")

        sparse_path.mkdir(parents=True, exist_ok=True)
        
        _stdout = stream_file if stream_file else subprocess.PIPE
        with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    print(line)
                    if line.startswith("Registering image #"):
                        line_process = line\
                            .replace("Registering image #", "")\
                            .replace("\n", "")
                        *_, current = line_process.split("(")
                        current, *_ = current.split(")")
                        progress.update(task, completed=int(current), refresh=True)

        progress.update(task, completed=int(total), refresh=True)

    return_code = process.returncode

    if return_code == 0:
        console.log('✅ Bundle adjustment completed.')
    else:
        raise FailedProcess("Bundle adjustment failed.")

def colmap_image_undistortion(
        image_path: Path,
        sparse0_path: Path,
        source_path: Path,
        colmap_command: str = "colmap",
        stream_file: Optional[IOBase] = None
    ):
    total = len(list(image_path.glob("*.jpg")))
    with Progress(console=console) as progress:
        task = progress.add_task("Image Undistortion", total=total)
        cmd = [
            colmap_command,
            "image_undistorter",
            "--image_path", image_path.as_posix(),
            "--input_path", sparse0_path.as_posix(),
            "--output_path", source_path.as_posix(),
            "--output_type", "COLMAP"
        ]
        console.log(f"💻 Executing command: {' '.join(cmd)}")

        _stdout = stream_file if stream_file else subprocess.PIPE
        with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    if line.startswith("Undistorting image ["):
                        line_process = line\
                            .replace("Undistorting image [", "")\
                            .replace("]", "")\
                            .replace("\n", "")
                        current, total = line_process.split("/")
                        progress.update(task, completed=int(current), total=int(total), refresh=True)

        progress.update(task, completed=int(total), refresh=True)

    return_code = process.returncode

    if return_code == 0:
        console.log('✅ Image undistortion completed.')
    else:
        raise FailedProcess("Image undistortion failed.")

def colmap(
    source_path: Path,
    camera: Literal["OPENCV"] = "OPENCV",
    colmap_command: str = "colmap",
    use_gpu: bool = True,
    skip_matching: bool = False,
    stream_file: Optional[IOBase] = None
):
    image_path = source_path / "input"
    if not image_path.exists():
        raise Exception(f"Image path {image_path} does not exist. Exiting.")

    total = len(list(image_path.glob("*.jpg")))
    if total == 0:
        raise Exception(f"No images found in {image_path}. Exiting.")

    database_path = source_path / "distorted" / "database.db"

    sparse_path = source_path / "distorted" / "sparse"

    if not skip_matching:
        colmap_feature_extraction(database_path, image_path, camera, colmap_command, use_gpu, stream_file)
        colmap_feature_matching(database_path, image_path, colmap_command, use_gpu, stream_file)
        colmap_bundle_adjustment(database_path, image_path, sparse_path, colmap_command, stream_file)

    colmap_image_undistortion(image_path, sparse_path / "0", source_path, colmap_command, stream_file)

    origin_path = source_path / "sparse"
    destination_path = source_path / "sparse" / "0"
    destination_path.mkdir(exist_ok=True)
    console.log(f"🌟 Moving files from {origin_path} to {destination_path}")
    for file in os.listdir(origin_path):
        if file == '0':
            continue
        source_file = os.path.join(origin_path, file)
        destination_file = os.path.join(destination_path, file)
        shutil.copy(source_file, destination_file)

if __name__ == "__main__":
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+t') as temp_file:
        print(f"Using temp file: {temp_file.name}")
        try:
            colmap(
                source_path = Path("/home/europe/Desktop/gaussian-splatting-kit/test/"),
                camera = "OPENCV",
                colmap_command = "colmap",
                use_gpu = True,
                skip_matching = False,
                stream_file = open("/home/europe/Desktop/gaussian-splatting-kit/test.log", "w+t")
            )
        except FailedProcess:
            console.log("🚨 Error executing colmap.")
            temp_file.seek(0)
            print(temp_file.read())