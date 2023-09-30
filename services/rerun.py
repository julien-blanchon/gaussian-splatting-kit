import os
import re
from pathlib import Path

import numpy as np
import rerun as rr  # pip install rerun-sdk
from utils.read_write_model import read_model

# From https://github.com/rerun-io/rerun/tree/main/examples/python/structure_from_motion
def read_and_log_sparse_reconstruction(
        exp_name: str,
        dataset_path: Path,
        output_path: Path,
        filter_output: bool = False,
        filter_min_visible: int = 2_000
    ) -> None:
    rr.init(exp_name)

    cameras, images, points3D = read_model(dataset_path / "sparse", ext=".bin")

    if filter_output:
        # Filter out noisy points
        points3D = {id: point for id, point in points3D.items() if point.rgb.any() and len(point.image_ids) > 4}

    rr.log_view_coordinates("/", up="-Y", timeless=True)

    # Iterate through images (video frames) logging data related to each frame.
    for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
        image_file = dataset_path / "images" / image.name

        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        np.array([1.0, 1.0])

        visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]
        visible_ids = image.point3D_ids[visible]

        if filter_output and len(visible_ids) < filter_min_visible:
            continue

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]

        rr.set_time_sequence("frame", frame_idx)

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        rr.log_scalar("plot/avg_reproj_err", np.mean(point_errors), color=[240, 45, 58])

        rr.log_points("points", points, colors=point_colors, ext={"error": point_errors})

        # COLMAP's camera transform is "camera from world"
        rr.log_transform3d(
            "camera", rr.TranslationRotationScale3D(image.tvec, rr.Quaternion(xyzw=quat_xyzw)), from_parent=True
        )
        rr.log_view_coordinates("camera", xyz="RDF")  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        assert camera.model == "PINHOLE"
        rr.log_pinhole(
            "camera/image",
            width=camera.width,
            height=camera.height,
            focal_length_px=camera.params[:2],
            principal_point_px=camera.params[2:],
        )

        rr.log_image_file("camera/image", img_path=dataset_path / "images" / image.name)
        rr.log_points("camera/image/keypoints", visible_xys, colors=[34, 138, 167])

    rerun_output_directory = output_path / "rerun" 
    rerun_output_directory.mkdir(parents=True, exist_ok=True)
    rerun_output_file = rerun_output_directory / "recording.rrd"
    rr.save(rerun_output_file.as_posix())

