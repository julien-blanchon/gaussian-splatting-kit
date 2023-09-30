from io import IOBase
from pathlib import Path
import subprocess
from typing import Optional
from rich.console import Console

console = Console()

def gaussian_splatting_cuda_training(
        data_path: Path,
        output_path: Path,
        gs_command: str,
        iterations: int = 10_000,
        convergence_rate: float = 0.01,
        resolution: int = 512,
        enable_cr_monitoring: bool = False,
        force: bool = False,
        empty_gpu_cache: bool = False,
        stream_file: Optional[IOBase] = None
    ) -> str:   
    """
    Core Options
    -h, --help
    Display this help menu.

    -d, --data_path [PATH]
    Specify the path to the training data.

    -f, --force
    Force overwriting of output folder. If not set, the program will exit if the output folder already exists.

    -o, --output_path [PATH]
    Specify the path to save the trained model. If this option is not specified, the trained model will be saved to the "output" folder located in the root directory of the project.

    -i, --iter [NUM]
    Specify the number of iterations to train the model. Although the paper sets the maximum number of iterations at 30k, you'll likely need far fewer. Starting with 6k or 7k iterations should yield preliminary results. Outputs are saved every 7k iterations and also at the end of the training. Therefore, even if you set it to 5k iterations, an output will be generated upon completion.

    Advanced Options
    --empty-gpu-cache Empty CUDA memory after ever 100 iterations. Attention! This has a considerable performance impact

    --enable-cr-monitoring
    Enable monitoring of the average convergence rate throughout training. If done, it will stop optimizing when the average convergence rate is below 0.008 per default after 15k iterations. This is useful for speeding up the training process when the gain starts to dimish. If not enabled, the training will stop after the specified number of iterations --iter. Otherwise its stops when max 30k iterations are reached.

    -c, --convergence_rate [RATE]
    Set custom average onvergence rate for the training process. Requires the flag --enable-cr-monitoring to be set.
    """ 
    
    cmd = [
        gs_command,
        f"--data-path={data_path.as_posix()}"
        f"--output-path={output_path.as_posix()}"
        f"--iter={iterations}",
        # TODO: Enable these options and put the right defaults in the function signature
        # f"--convergence-rate={convergence_rate}",
        # f"--resolution={resolution}",
        # "--enable-cr-monitoring" if enable_cr_monitoring else "",
        # "--force" if force else "",
        # "--empty-gpu-cache" if empty_gpu_cache else ""
    ]

    console.log(f"💻 Executing command: {' '.join(cmd)}")

    _stdout = stream_file if stream_file else subprocess.PIPE
    with subprocess.Popen(cmd, stdout=_stdout, stderr=subprocess.STDOUT, text=True) as process:
        if process.stdout:
            for line in process.stdout:
                print(line)
                

    # Check if the command was successful
    return_code = process.returncode
    if return_code == 0:
        console.log('✅ Successfully splatted frames.')
    else:
        raise Exception('Error splatting frames.')
        
def gaussian_splatting_cuda(
        data_path: Path,
        output_path: Path,
        gs_command: str,
        iterations: int = 10_000,
        convergence_rate: float = 0.01,
        resolution: int = 512,
        enable_cr_monitoring: bool = False,
        force: bool = False,
        empty_gpu_cache: bool = False,
        stream_file: Optional[IOBase] = None
    ) -> str: 
    # Check if the output path exists
    if output_path.exists() and not force:
        raise Exception(f"Output folder already exists. Path: {output_path}, use --force to overwrite.")

    # Create the output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Execute gaussian_splatting_cuda
    gaussian_splatting_cuda_training(
        data_path,
        output_path,
        gs_command,
        iterations,
        convergence_rate,
        resolution,
        enable_cr_monitoring,
        force,
        empty_gpu_cache,
        stream_file
    )