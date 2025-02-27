# main.py

import cv2
import numpy as np
from audio import AudioGenerator
from cv2display import CV2Display
from lidar import LidarDevice
from pointcloud import PointCloudProcessor


def main(dev_idx: int = 0) -> None:
    # Initialize LiDAR device
    lidar = LidarDevice()
    lidar.connect(dev_idx)

    # Initialize point cloud processing (for video display)
    pc_processor = PointCloudProcessor()

    # Initialize CV2 display
    display = CV2Display()

    # Wait for the first frame and set up internal parameters
    lidar.event.wait()
    pc_processor.setup_intrinsics(lidar.session)
    pc_processor.pcd = pc_processor.create_point_cloud(lidar.session)
    pc_processor.vis.add_geometry(pc_processor.pcd)

    # Initialize audio generator
    audio_gen = AudioGenerator()

    while True:
        lidar.event.wait()
        rgb_frame = lidar.session.get_rgb_frame()
        depth_frame = lidar.session.get_depth_frame()

        # Draw video overlay with CV2
        overlay = display.draw_overlay(rgb_frame, depth_frame)

        # Point cloud processing (for video)
        new_pcd = pc_processor.create_point_cloud(lidar.session)
        processed_pcd = pc_processor.process_point_cloud(new_pcd)
        pc_processor.record_frame()
        pc_processor.update_visualization(processed_pcd)

        # Generate audio feedback using the central value of depth_frame
        height, width = depth_frame.shape
        depth_center = depth_frame[height // 2, width // 2]

        # Use non-linear mapping with exponent=2 (adjust as needed)
        volume = audio_gen.map_depth_to_volume(
            depth_center, max_distance=2.0, exponent=1
        )

        # Generate a tone with a fixed frequency (e.g., 440Hz) (play the same for left and right)
        audio_gen.play_tone(440, volume)

        display.show(overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        lidar.event.clear()

    # Cleanup
    display.close()
    pc_processor.vis.destroy_window()
    audio_gen.close()


if __name__ == "__main__":
    main(dev_idx=0)
