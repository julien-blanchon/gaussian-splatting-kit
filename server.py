from pathlib import Path
import shutil
import tempfile
import gradio as gr
import uuid
from typing_extensions import TypedDict, Tuple

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# create a static directory to store the static files
gs_dir = Path(str(tempfile.gettempdir())) / "gaussian_splatting_gradio"
gs_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=gs_dir), name="static")

StateDict = TypedDict("StateDict", {
    "uuid": str,
})

def getHTML():
    html_body = """
<body>
    <div id="progress"></div>
    <div id="message"></div>
    <div class="scene" id="spinner">
        <div class="cube-wrapper">
            <div class="cube">
                <div class="cube-faces">
                    <div class="cube-face bottom"></div>
                    <div class="cube-face top"></div>
                    <div class="cube-face left"></div>
                    <div class="cube-face right"></div>
                    <div class="cube-face back"></div>
                    <div class="cube-face front"></div>
                </div>
            </div>
        </div>
    </div>
    <canvas id="canvas"></canvas>

    <div id="quality">
        <span id="fps"></span>
    </div>

    <style>
        .cube-wrapper {
            transform-style: preserve-3d;
        }

        .cube {
            transform-style: preserve-3d;
            transform: rotateX(45deg) rotateZ(45deg);
            animation: rotation 2s infinite;
        }

        .cube-faces {
            transform-style: preserve-3d;
            height: 80px;
            width: 80px;
            position: relative;
            transform-origin: 0 0;
            transform: translateX(0) translateY(0) translateZ(-40px);
        }

        .cube-face {
            position: absolute;
            inset: 0;
            background: #0017ff;
            border: solid 1px #ffffff;
        }
        .cube-face.top {
            transform: translateZ(80px);
        }
        .cube-face.front {
            transform-origin: 0 50%;
            transform: rotateY(-90deg);
        }
        .cube-face.back {
            transform-origin: 0 50%;
            transform: rotateY(-90deg) translateZ(-80px);
        }
        .cube-face.right {
            transform-origin: 50% 0;
            transform: rotateX(-90deg) translateY(-80px);
        }
        .cube-face.left {
            transform-origin: 50% 0;
            transform: rotateX(-90deg) translateY(-80px) translateZ(80px);
        }

        @keyframes rotation {
            0% {
                transform: rotateX(45deg) rotateY(0) rotateZ(45deg);
                animation-timing-function: cubic-bezier(
                    0.17,
                    0.84,
                    0.44,
                    1
                );
            }
            50% {
                transform: rotateX(45deg) rotateY(0) rotateZ(225deg);
                animation-timing-function: cubic-bezier(
                    0.76,
                    0.05,
                    0.86,
                    0.06
                );
            }
            100% {
                transform: rotateX(45deg) rotateY(0) rotateZ(405deg);
                animation-timing-function: cubic-bezier(
                    0.17,
                    0.84,
                    0.44,
                    1
                );
            }
        }

        .scene,
        #message {
            position: absolute;
            display: flex;
            top: 0;
            right: 0;
            left: 0;
            bottom: 0;
            z-index: 2;
            height: 100%;
            width: 100%;
            align-items: center;
            justify-content: center;
        }
        #message {
            font-weight: bold;
            font-size: large;
            color: red;
            pointer-events: none;
        }

        #progress {
            position: absolute;
            top: 0;
            height: 5px;
            background: blue;
            z-index: 99;
            transition: width 0.1s ease-in-out;
        }

        #quality {
            position: absolute;
            bottom: 10px;
            z-index: 999;
            right: 10px;
        }

        #canvas {
            display: block;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            touch-action: none;
        }

        #instructions {
            background: rgba(0,0,0,0.6);
            white-space: pre-wrap;
            padding: 10px;
            border-radius: 10px;
            font-size: x-small;
        }
    </style>
</body>
"""

    html = f"""
<head>
  <title>3D Gaussian Splatting Viewer</title>
  <script src="http://zeus.blanchon.cc/dropshare/main.js"></script>
</head>

{html_body}
"""
    return f"""<iframe style="width: 100%; height: 900px" srcdoc='{html}'></iframe>"""

def createStateSession() -> StateDict:
    # Create new session
    session_uuid = str(uuid.uuid4())
    print("createStateSession")
    print(session_uuid)
    return StateDict(
        uuid=session_uuid,
    )

def removeStateSession(session_state_value: StateDict):
    # Clean up previous session
    return StateDict(
        uuid=None,
    )

def makeButtonVisible() -> Tuple[gr.Button, gr.Button]:
    process_button = gr.Button(visible=True)
    reset_button = gr.Button(visible=False) #TODO: I will bring this back when I figure out how to stop the process
    return process_button, reset_button
    
def resetSession(state: StateDict) -> Tuple[StateDict, gr.Button, gr.Button]:
    print("resetSession")
    new_state = removeStateSession(state)
    process_button = gr.Button(visible=False)
    reset_button = gr.Button(visible=False)
    return new_state, process_button, reset_button

def process(
        # *args, **kwargs
        session_state_value: StateDict,
        filepath: str,
        ffmpeg_fps: int,
        ffmpeg_qscale: int,
        colmap_camera: str,
    ):
    if session_state_value["uuid"] is None:
        return
    print("process")
    # print(args)
    # print(kwargs)
    # return
    print(session_state_value)
    print(f"Processing {filepath}")

    try:
        session_tmpdirname = gs_dir / str(session_state_value['uuid'])
        session_tmpdirname.mkdir(parents=True, exist_ok=True)
        print('Created temporary directory', session_tmpdirname)

        gs_dir_path = Path(session_tmpdirname)
        logfile_path = Path(session_tmpdirname) / "log.txt"
        logfile_path.touch()
        with logfile_path.open("w") as log_file:
            # Create log file
            logfile_path.touch()

            from services.ffmpeg import ffmpeg_run
            ffmpeg_run(
                video_path = Path(filepath),
                output_path = gs_dir_path,
                fps = int(ffmpeg_fps),
                qscale = int(ffmpeg_qscale),
                stream_file=log_file
            )

            from services.colmap import colmap
            colmap(
                source_path=gs_dir_path,
                camera=str(colmap_camera),
                stream_file=log_file
            )

            print("Done with colmap")

            # Create a zip of the gs_dir_path folder
            print(gs_dir, gs_dir_path)
            print(gs_dir_path.name)
            archive = shutil.make_archive("result", 'zip', gs_dir, gs_dir_path)
            print('Created zip file', archive)

            # Move the zip file to the gs_dir_path folder
            shutil.move(archive, gs_dir_path)

            from services.gaussian_splatting_cuda import gaussian_splatting_cuda
            gaussian_splatting_cuda(
                data_path = gs_dir_path,
                output_path = gs_dir_path / "output",
                gs_command = str(Path(__file__).parent.absolute() / "build" / 'gaussian_splatting_cuda'),
                iterations = 100,
                convergence_rate = 0.01,
                resolution = 512,
                enable_cr_monitoring = False,
                force = False,
                empty_gpu_cache = False,
                stream_file = log_file
            )

    except Exception:
        pass
        # print('Error - Removing temporary directory', session_tmpdirname)
        # shutil.rmtree(session_tmpdirname)

def updateLog(session_state_value: StateDict) -> str:
    if session_state_value["uuid"] is None:
        return ""

    log_file = gs_dir / str(session_state_value['uuid']) / "log.txt"
    if not log_file.exists():
        return ""
    
    with log_file.open("r") as log_file:
        logs = log_file.read()

    return logs

with gr.Blocks() as demo:
    session_state = gr.State({
        "uuid": None,
    })

    with gr.Row():

        with gr.Column():
            video_input = gr.PlayableVideo(
                format="mp4",
                source="upload",
                label="Upload a video",
                include_audio=False
            )
            with gr.Row(variant="panel"):
                ffmpeg_fps = gr.Number(
                    label="FFMPEG FPE",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=0.10,
                )
                ffmpeg_qscale = gr.Number(
                    label="FFMPEG QSCALE",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=1,
                )
                colmap_camera = gr.Dropdown(
                    label="COLMAP Camera",
                    value="OPENCV",
                    choices=["OPENCV", "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL"],
                )            

        text_log = gr.Textbox(
            label="Logs",
            info="Logs",
            interactive=False,
            show_copy_button=True
        )
        # text_log = gr.Code(
        #     label="Logs",
        #     language=None,
        #     interactive=False,
        # )
    

    process_button = gr.Button("Process", visible=False)
    reset_button = gr.ClearButton(
        components=[video_input, text_log, ffmpeg_fps, ffmpeg_qscale, colmap_camera],
        label="Reset",
        visible=False,
    )

    process_event = process_button.click(
        fn=process,
        inputs=[session_state, video_input, ffmpeg_fps, ffmpeg_qscale, colmap_camera],
        outputs=[],
    )

    upload_event = video_input.upload(
        fn=makeButtonVisible,
        inputs=[],
        outputs=[process_button, reset_button]
    ).then(
        fn=createStateSession,
        inputs=[],
        outputs=[session_state],
    ).then(
        fn=updateLog,
        inputs=[session_state],
        outputs=[text_log],
        every=2,
    )

    reset_button.click(
        fn=resetSession,
        inputs=[session_state],
        outputs=[session_state, process_button, reset_button],
        cancels=[process_event]
    )
    
    video_input.clear(
        fn=resetSession,
        inputs=[session_state],
        outputs=[session_state, process_button, reset_button],
        cancels=[process_event]
    )

    demo.close


    # gr.LoginButton, gr.LogoutButton
    # gr.HuggingFaceDatasetSaver
    # gr.OAuthProfile
    
    

    
    

    # with gr.Tab("jsdn"):
    #     input_mic = gr.HTML(getHTML())

demo.queue()
# demo.launch()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
