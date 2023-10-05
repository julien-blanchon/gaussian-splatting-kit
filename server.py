import os
from pathlib import Path
import shutil
import tempfile
from typing import List
import gradio as gr
import uuid
from typing_extensions import TypedDict, Tuple

from fastapi import FastAPI

app = FastAPI()

# create a static directory to store the static files
GS_DIR = Path(str(tempfile.gettempdir())) / "gaussian_splatting_gradio"
GS_DIR.mkdir(parents=True, exist_ok=True)

StateDict = TypedDict("StateDict", {
    "uuid": str,
})

# http://localhost:7860/file=/tmp/gradio/c2110a7de804b39754d229de426dc9307bc03aea/page.svelte

HOST = "localhost"
PORT = 7860

home_markdown = """
...
"""

step1_markdown = """
# Step 1 - Split Video into Frames

In the journey of transforming a video into a 3D Gaussian Splatting, the initial step is the conversion of the video into individual frames. You can here provide a **video file** and specify how much image you want to extract per second (*fps*). The application will then automatically extract the frames from the video and prepare them for the next step in the process.

However, you can also do this step manually and upload the frames directory by yourself in the next step. In this case, you can skip this step and go directly to the next step.

Please not that blurry frames will mostlikely result in a bad 3D model. So, make sure that the video is clear enough.
"""

step2_markdown = """
# Step 2 - SfM using Colmap

In this step we use Colmap (https://github.com/colmap/colmap). This process utilizes the frames extracted from the uploaded video to generate camera parameters and a point cloud, which are essential components for the 3D Gaussian Splatting process.

This step could take a while depending on the number of frames and the resolution. So, please be patient. 
You might want to do this step manually and upload the frames directory by yourself in the next step. In this case, you can skip this step and go directly to the next step.
"""

step3_markdown = """
# Step 3 - 3D Gaussian Splatting

In this final step we use the 3D Gaussian Splatting Cuda implementation by MrNeRF (https://twitter.com/janusch_patas): https://github.com/MrNeRF/gaussian-splatting-cuda.
As it's quite rapid to train, you can easily use a high number of iterations.
"""

def getPlyFile(session_state_value: StateDict) -> str:
    return f"/tmp/gaussian_splatting_gradio/{session_state_value['uuid']}/output/final_point_cloud.ply"

def getCamerasFile(session_state_value: StateDict) -> str:
    return f"/tmp/gaussian_splatting_gradio/{session_state_value['uuid']}/output/cameras.json"

def getZipFile(session_state_value: StateDict) -> str:
    return f"/tmp/gaussian_splatting_gradio/{session_state_value['uuid']}/result.zip"

def makeResult(session_state_value: StateDict) -> tuple[str, str, str]:
    ply_file = getPlyFile(session_state_value)
    cameras_file = getCamerasFile(session_state_value)
    zip_file = getZipFile(session_state_value)
    return [ply_file, cameras_file, zip_file]


# Utility functions
def createStateSession(previous_session: StateDict) -> StateDict:
    if previous_session["uuid"] is None:
        # Create new session
        session_uuid = str(uuid.uuid4())
        print("Creating new session: ", session_uuid)
        session_tmpdirname = GS_DIR / str(session_uuid)
        session_tmpdirname.mkdir(parents=True, exist_ok=True)
        print('Created temporary directory: ', session_tmpdirname)
        session = StateDict(
            uuid=session_uuid,
        )
    else:
        # Use previous session
        session = previous_session
    return session

def removeStateSession(session_state_value: StateDict):
    # Clean up previous session
    session_uuid = session_state_value["uuid"]
    session_tmpdirname = GS_DIR / str(session_uuid)
    print('Removing temporary directory: ', session_tmpdirname)
    shutil.rmtree(session_tmpdirname)
    return StateDict(
        uuid=None,
    )

def makeButtonVisible(btn_value: str) -> gr.Button:
    return gr.Button(btn_value, visible=True)


#  Process functions
def process_ffmpeg(
        session_state_value: StateDict,
        ffmpeg_input: str,
        ffmpeg_fps: int,
        ffmpeg_qscale: int,
    ) -> list[str]:
    # Ensure that a session is active
    if session_state_value["uuid"] is None:
        return

    # Set up session directory
    session_path = GS_DIR / str(session_state_value['uuid'])
    logfile_path = Path(session_path) / "ffmpeg_log.txt"
    logfile_path.touch()

    try:
        from services.ffmpeg import ffmpeg_run
        with logfile_path.open("w") as log_file:
            ffmpeg_run(
                video_path = Path(ffmpeg_input),
                output_path = session_path,
                fps = int(ffmpeg_fps),
                qscale = int(ffmpeg_qscale),
                stream_file=log_file
            )
        print("Done with ffmpeg")
    except Exception as e:
        print(f"Error - {e}")
        # print('Error - Removing temporary directory', session_path)
        # shutil.rmtree(session_path)
    # Get the list of all the file of (session_path / "input")
    list_of_jpgs = [str(f) for f in (session_path / "input").glob("*.jpg")]
    return list_of_jpgs

def processColmap(
        session_state_value: StateDict,
        colmap_inputs: List[tempfile.NamedTemporaryFile],
        colmap_camera: str,
        enable_rerun: bool
    ) -> Tuple[str, str]:
    # Ensure that a session is active
    if session_state_value["uuid"] is None:
        return "", ""
        
    # Set up session directory
    session_path = GS_DIR / str(session_state_value['uuid'])
    logfile_path = Path(session_path) / "colmap_log.txt"
    logfile_path.touch()

    rerunfile_path = Path(session_path) / "rerun_page.html"
    rerunfile_path.touch()

    (session_path / "input").mkdir(parents=True, exist_ok=True)
    for file in colmap_inputs:
        print("copying", file.name, "to", session_path / "input")
        shutil.copy(file.name, session_path / "input")

    try:
        from services.colmap import colmap
        with logfile_path.open("w") as log_file:
            colmap(
                source_path=session_path,
                camera=str(colmap_camera),
                stream_file=log_file
            )
        print("Done with colmap")
        
        if enable_rerun:
            from services.rerun import read_and_log_sparse_reconstruction
            html = read_and_log_sparse_reconstruction(
                exp_name = str(session_state_value['uuid']),
                dataset_path = session_path,
            )
            print("Done with rerun")
        else:
            html = "Rerun was disable !"
        with rerunfile_path.open("w") as rerunfile:
            rerunfile.write(html)
    except Exception as e:
        print(f"Error - {e}")
        # print('Error - Removing temporary directory', session_path)
        # shutil.rmtree(session_path)

    # zip the session_path folder
    archive = shutil.make_archive("result", 'zip', GS_DIR, session_path)
    print('Created zip file', archive)
    return archive, rerunfile_path

def processGaussianSplattingCuda(
        session_state_value: StateDict,
        gs_input: tempfile.NamedTemporaryFile,
        gs_iterations: int,
        gs_convergence_rate: float,
        gs_resolution: int,
    ) -> Tuple[str, str]:
    # Ensure that a session is active
    if session_state_value["uuid"] is None:
        return
    
    # Set up session directory
    session_path = GS_DIR / str(session_state_value['uuid'])
    logfile_path = Path(session_path) / "gaussian_splatting_cuda_log.txt"
    logfile_path.touch()

    # Unzip the gs_input file to the session_path
    shutil.unpack_archive(gs_input.name, session_path)

    # Copy the gs_input directory to the session_path
    # shutil.copytree(gs_input, session_path)

    try:
        from services.gaussian_splatting_cuda import gaussian_splatting_cuda
        with logfile_path.open("w") as log_file:
            gaussian_splatting_cuda(
                data_path = session_path,
                output_path = session_path / "output",
                gs_command = str(Path(__file__).parent.absolute() / "build" / 'gaussian_splatting_cuda'),
                iterations = int(gs_iterations),
                convergence_rate = float(gs_convergence_rate),
                resolution = int(gs_resolution),
                enable_cr_monitoring = False,
                force = False,
                empty_gpu_cache = False,
                stream_file = log_file
            )
        print("Done with gaussian_splatting_cuda")

        # Create a zip of the session_path folder
        archive = shutil.make_archive("result", 'zip', GS_DIR, session_path)
        print('Created zip file', archive)

        # Move the zip file to the session_path folder
        shutil.move(archive, session_path)
    except Exception as e:
        print(f"Error - {e}")
        # print('Error - Removing temporary directory', session_path)
        # shutil.rmtree(session_path)
    
    return (
        session_path / "output" / "final_point_cloud.ply",
        session_path / "output" / "cameras.json",
    )

def updateLog(logname:str, session_state_value: StateDict) -> str:
    if session_state_value["uuid"] is None:
        return ""

    log_file = GS_DIR / str(session_state_value['uuid']) / f"{logname}.txt"
    if not log_file.exists():
        return ""
    
    with log_file.open("r") as log_file:
        logs = log_file.read()

    return logs

def bindStep1Step2(step1_output: list[tempfile.NamedTemporaryFile]) -> list[str]:
    return [file.name for file in step1_output]

def bindStep2Step3(step2_output: tempfile.NamedTemporaryFile) -> str:
    return step2_output.name

def makeRerunIframe(rerun_html : tempfile.NamedTemporaryFile) -> str:
    # If rerun_html is bigger than 300MB, then we don't show it
    print(f"Rerun file size: {os.stat(rerun_html.name).st_size}")
    if os.stat(rerun_html.name).st_size > 100_000_000:
        print("Rerun file is too big, not showing it")
        return ""
    filepath = rerun_html.name
    print("filepath", filepath)
    return f"""<iframe src="/file={filepath}" width="100%"; height="1080px"></iframe>"""

with gr.Blocks() as demo:
    #############################
    ########## State ############
    #############################

    session_state = gr.State({
        "uuid": None,
    })

    #############################
    ###### UI Components ########
    #############################

    gr.Markdown("# Gaussian Splatting Kit")
    gr.Markdown("Click on the **Duplicate** button to create a new instance of this app.")
    duplicate_button = gr.DuplicateButton()
    gr.Markdown(value=home_markdown)

    with gr.Tab("Slit Video into Frames"):
        step1_description = gr.Markdown(step1_markdown)
        # Video Frames
        with gr.Row():   
            # Video Frames - Inputs
            with gr.Column():
                # Video Frames - Inputs - Video File
                step1_input = gr.PlayableVideo(
                    format="mp4",
                    source="upload",
                    label="Upload a video",
                    include_audio=False
                )
                # Video Frames - Inputs - Parameters
                with gr.Row(variant="panel"):
                    # Video Frames - Inputs - Parameters - FFMPEG FPS
                    step1_fps = gr.Number(
                        label="FFMPEG Fps",
                        value=1,
                        minimum=1,
                        maximum=5,
                        step=0.10,
                    )
                    # Video Frames - Inputs - Parameters - FFMPEG Qscale
                    step1_qscale = gr.Number(
                        label="FFMPEG Qscale",
                        value=1,
                        minimum=1,
                        maximum=5,
                        step=1,
                    )
            # Video Frames - Outputs
            with gr.Column():
                # Video Frames - Outputs - Video File
                step1_output = gr.File(
                    label="Frames",
                    file_count="directory",
                    type="file",
                    interactive=False,
                )
                # Video Frames - Outputs - Logs
                step1_logs = gr.Textbox(
                    label="Videos Logs",
                    interactive=False,
                    show_copy_button=True
                )
        # Video Frames - Process Button
        step1_processbtn = gr.Button("Process", visible=True)
        # Video Frames - Visualize
        # Video Frames - Visualize -
        # step1_visualize_gallery = gr.Gallery()
             
    with gr.Tab("Colmap"):
        step2_description = gr.Markdown(step2_markdown)
        # Colmap
        with gr.Row():
            # Colmap - Inputs
            with gr.Column():
                # Colmap - Inputs - Frames Directory
                step2_input = gr.File(
                    label="Upload a frames directory",
                    file_count="directory",
                    type="file",
                    interactive=True,
                )
                # Colmap - Inputs - Parameters
                with gr.Row(variant="panel"):
                    # Colmap - Inputs - Parameters - Colmap Camera
                    step2_camera = gr.Dropdown(
                        label="COLMAP Camera",
                        value="OPENCV",
                        choices=["OPENCV", "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL"],
                    )
                    # Colmap - Inputs - Parameters - Enable Rerun
                    step2_rerun = gr.Checkbox(
                        value=True,
                        label="Enable Rerun",
                    )
            # Colmap - Outputs
            with gr.Column():
                # Colmap - Outputs - Video File
                step2_output = gr.File(
                    label="Colmap",
                    file_count="single",
                    file_types=[".zip"],
                    type="file",
                    interactive=False,
                )
                # Colmap - Outputs - Logs
                step2_logs = gr.Textbox(
                    label="Colmap Logs",
                    interactive=False,
                    show_copy_button=True
                )
                
        # Colmap - Process Button
        step2_processbtn = gr.Button("Process", visible=True)

        # Colmap - Visualize
        # Colmap - Visualize - Rerun HTML File
        step_2_visualize_html = gr.File(
            label="Rerun HTML",
            file_count="single",
            file_types=[".html"],
            type="file",
            interactive=False,
            visible=False
        )
        # Colmap - Visualize - Rerun HTML
        step_2_visualize = gr.HTML("Rerun", visible=True)
    
    with gr.Tab("Gaussian Splatting"):
        step3_description = gr.Markdown(step3_markdown)
        # Gaussian Splatting
        with gr.Row():
            # Gaussian Splatting - Inputs
            with gr.Column():
                # Gaussian Splatting - Inputs - Colmap + Frames
                step3_input = gr.File(
                    label="Upload a colmap + frames directory",
                    file_count="single",
                    file_types=[".zip"],
                    type="file",
                    interactive=True,
                )
                # Gaussian Splatting - Inputs - Parameters
                with gr.Row(variant="panel"):
                    # Gaussian Splatting - Inputs - Parameters - GS Iterations
                    step3_iterations = gr.Number(
                        label="GS Iterations",
                        value=10_000,
                        minimum=1_000,
                        maximum=50_000,
                        step=1_000,
                    )
                    # Gaussian Splatting - Inputs - Parameters - GS Convergence Rate
                    step3_convergence_rate = gr.Number(
                        label="GS Convergence Rate",
                        value=0.01,
                        minimum=0.01,
                        maximum=1,
                        step=0.01,
                    )
                    # Gaussian Splatting - Inputs - Parameters - GS Resolution
                    step3_resolution = gr.Number(
                        label="GS Resolution",
                        value=512,
                        minimum=128,
                        maximum=1024,
                        step=128,
                    )
            # Gaussian Splatting - Outputs
            with gr.Column():
                with gr.Row():
                    # Gaussian Splatting - Outputs - PLY File
                    step3_output1 = gr.File(
                        label="PLY File",
                        file_count="single",
                        type="file",
                        interactive=False,
                    )
                
                    # Gaussian Splatting - Outputs - Cameras File
                    step3_output2 = gr.File(
                        label="Cameras File",
                        file_count="single",
                        type="file",
                        interactive=False,
                    )
                # Gaussian Splatting - Outputs - Logs
                step3_logs = gr.Textbox(
                    label="Gaussian Splatting Logs",
                    interactive=False,
                    show_copy_button=True
                )
        # Gaussian Splatting - Process Button
        step3_processbtn = gr.Button("Process", visible=True)
        # Gaussian Splatting - Visualize
        # Gaussian Splatting - Visualize - Antimatter15 HTML
        # step_3_visualize = gr.HTML(getAntimatter15HTML(), visible=True)
        step_3_visualize = gr.Button("Visualize", visible=True, link="https://antimatter15.com/splat/")

    #############################
    ########## Events ###########
    #############################
    ### Step 1
    # Make the process button visible when a video is uploaded
    step1_upload_event = step1_input.upload(
        fn=createStateSession,
        inputs=[session_state],
        outputs=[session_state]
    ).success(
        fn=makeButtonVisible,
        inputs=[step1_processbtn],
        outputs=[step1_processbtn],
    )
    # Do the processing when the process button is clicked
    step1_processevent = step1_processbtn.click(
        fn=process_ffmpeg,
        inputs=[session_state, step1_input, step1_fps, step1_qscale],
        outputs=[step1_output],
    ).success(
        fn=bindStep1Step2,
        inputs=[step1_output],
        outputs=[step2_input],
    ).success(
        fn=makeButtonVisible,
        inputs=[step2_processbtn],
        outputs=[step2_processbtn],
    )

    # Update the logs every 2 seconds
    step1_logsevent = step1_processbtn.click(
        fn=lambda session: updateLog("ffmpeg_log", session),
        inputs=[session_state],
        outputs=[step1_logs],
        every=2,
    )
    
    ## Step 2
    # Make the process button visible when a video is uploaded
    step2_upload_event = step2_input.upload(
        fn=createStateSession,
        inputs=[session_state],
        outputs=[session_state]
    ).success(
        fn=makeButtonVisible,
        inputs=[step2_processbtn],
        outputs=[step2_processbtn],
    )
    # Do the processing when the process button is clicked
    step2_processevent = step2_processbtn.click(
        fn=processColmap,
        inputs=[session_state, step2_input, step2_camera, step2_rerun],
        outputs=[step2_output, step_2_visualize_html]
    ).success(
        fn=bindStep2Step3,
        inputs=[step2_output],
        outputs=[step3_input],
    ).success(
        fn=makeButtonVisible,
        inputs=[step3_processbtn],
        outputs=[step3_processbtn],
    ).then(
        fn=makeRerunIframe,
        inputs=[step_2_visualize_html],
        outputs=[step_2_visualize],
    )

    # Update the logs every 2 seconds
    step2_logsevent = step2_processbtn.click(
        fn=lambda session: updateLog("colmap_log", session),
        inputs=[session_state],
        outputs=[step2_logs],
        every=2,
    )

    ## Step 3
    # Make the process button visible when a video is uploaded
    step3_upload_event = step3_input.upload(
        fn=createStateSession,
        inputs=[session_state],
        outputs=[session_state]
    ).success(
        fn=makeButtonVisible,
        inputs=[step3_processbtn],
        outputs=[step3_processbtn],
    )
    # Do the processing when the process button is clicked
    step3_processevent = step3_processbtn.click(
        fn=processGaussianSplattingCuda,
        inputs=[session_state, step3_input, step3_iterations, step3_convergence_rate, step3_resolution],
        outputs=[step3_output1, step3_output2]
    )
    # .success(
    #     fn=lambda x: x,
    #     inputs=[step3_output1, step3_output2],
    #     outputs=[],
    # )
    # Update the logs every 2 seconds
    step3_logsevent = step3_processbtn.click(
        fn=lambda session: updateLog("gaussian_splatting_cuda_log", session),
        inputs=[session_state],
        outputs=[step3_logs],
        every=2,
    )

    # reset_button = gr.ClearButton(
    #     components=[video_input, text_log, ffmpeg_fps, ffmpeg_qscale, colmap_camera],
    #     label="Reset",
    #     visible=False,
    # )
    # print(f"async (x) => {{ {getJS(url='http://0.0.0.0:7860/output/37c7ae54-7752-4e7b-8ba9-bab32c86b316/output/point_cloud/iteration_100/point_cloud.ply')} }}")
    
    # show_button.click(
    #     fn=None,
    #     inputs=[],
    #     outputs=[],
    #     _js=f"async (x) => {{ {getJS(url='http://0.0.0.0:7860/output/37c7ae54-7752-4e7b-8ba9-bab32c86b316/output/point_cloud/iteration_100/point_cloud.ply')} }}"
    # ).then(
    #     fn=None,
    #     inputs=[],
    #     outputs=[],
    #     _js=f"async (x) => {{ {getJS(url='http://0.0.0.0:7860/output/37c7ae54-7752-4e7b-8ba9-bab32c86b316/output/point_cloud/iteration_100/point_cloud.ply')} }}"
    # )

    # gr.LoginButton, gr.LogoutButton
    # gr.HuggingFaceDatasetSaver
    # gr.OAuthProfile

    # with gr.Tab("jsdn"):
    #     input_mic = gr.HTML(getRerunHTML())




demo.queue()
demo.launch()

# mount Gradio app to FastAPI app
# app = gr.mount_gradio_app(app, demo, path="/")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860, ws_max_size=16777216*1000)
