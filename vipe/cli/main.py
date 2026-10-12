# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing image frames",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
@click.option("--start_frame", type=int, default=0, help="Starting frame number (default: 0)")
@click.option("--end_frame", type=int, default=None, help="Ending frame number (inclusive, default: process all frames)")
@click.option("--assume_fixed_camera_pose", is_flag=True, help="Assume camera pose is fixed throughout the video (skips SLAM pose estimation)")
@click.option("--use_exo_intrinsic_gt", type=str, default=None, help="3x3 intrinsics matrix in JSON format, e.g., '[[fx,0,cx],[0,fy,cy],[0,0,1]]' (sets optimize_intrinsics=False)")
def infer(video: Path, image_dir: Path, output: Path, pipeline: str, visualize: bool, start_frame: int, end_frame: int, assume_fixed_camera_pose: bool, use_exo_intrinsic_gt: str):
    """Run inference on a video file or directory of images."""

    logger = configure_logging()

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file or --image-dir", err=True)
        raise click.Abort()
    
    if video and image_dir:
        click.echo("Error: Cannot provide both video file and --image-dir", err=True)
        raise click.Abort()

    # Create output directory based on parent folder name (video_name)
    # For structure like: {video_name}/exo.mp4 -> use {video_name}
    video_name = video.parent.name  # Get parent directory name
    video_output_path = output / video_name
    
    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={video_output_path}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")
    
    if assume_fixed_camera_pose:
        overrides.append("pipeline.assume_fixed_camera_pose=true")
        logger.info("Fixed camera pose mode enabled - SLAM pose estimation will be skipped")
    
    if use_exo_intrinsic_gt is not None:
        import json
        try:
            intrinsics_matrix = json.loads(use_exo_intrinsic_gt)
            if not isinstance(intrinsics_matrix, list) or len(intrinsics_matrix) != 3:
                raise ValueError("Intrinsics matrix must be a 3x3 list")
            for row in intrinsics_matrix:
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError("Intrinsics matrix must be a 3x3 list")
        except (json.JSONDecodeError, ValueError) as e:
            click.echo(f"Error: Invalid intrinsics matrix format: {e}", err=True)
            raise click.Abort()
        
        overrides.append("pipeline.slam.optimize_intrinsics=false")
        overrides.append(f"+pipeline.use_exo_intrinsic_gt={use_exo_intrinsic_gt}")
        logger.info(f"Exo GT intrinsics mode enabled with provided matrix - intrinsics optimization will be disabled")

    # Set up stream configuration based on input type
    if image_dir:
        overrides.extend([
            "streams=frame_dir_stream",
            f"streams.base_path={image_dir}"
        ])
        input_path = image_dir
        input_desc = f"image directory {image_dir}"
    else:
        input_path = video
        input_desc = f"video {video}"

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {input_desc}...")
    logger.info(f"Output will be saved to: {video_output_path}")
    vipe_pipeline = make_pipeline(args.pipeline)

    if image_dir:
        # Use frame directory stream
        video_stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(desc="Reading image frames")
    else:
        # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
        # Apply frame range if specified
        if end_frame is not None:
            seek_range = range(start_frame, end_frame + 1)  # +1 to make end_frame inclusive
            video_stream = ProcessedVideoStream(RawMp4Stream(video, seek_range=seek_range), []).cache(desc="Reading video stream")
            logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
        elif start_frame > 0:
            # If only start_frame is specified, process from start_frame to end
            video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")
            total_frames = len(video_stream)
            seek_range = range(start_frame, total_frames)
            video_stream = ProcessedVideoStream(RawMp4Stream(video, seek_range=seek_range), []).cache(desc="Reading video stream")
            logger.info(f"Processing frames {start_frame} to {total_frames-1} ({total_frames - start_frame} frames)")
        else:
            video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")
            logger.info(f"Processing all {len(video_stream)} frames (0 to {len(video_stream)-1})")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
@click.option("--use_mean_bg", is_flag=True, help="Use robust statistical mean background instead of standard background")
@click.option("--ego_manual", is_flag=True, help="Enable manual ego camera control with transform handles")
@click.option("--frame_fix", is_flag=True, help="Fix the number of frames to 49")
def visualize(data_path: Path, port: int, use_mean_bg: bool, ego_manual: bool, frame_fix: bool):
    run_viser(data_path, port, use_mean_bg, ego_manual, frame_fix)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
