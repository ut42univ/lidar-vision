# main.py

import cv2
import numpy as np
import pyaudio
from cv2display import CV2Display
from lidar import LidarDevice
from pointcloud import PointCloudProcessor

# Function for generating sound
SAMPLE_RATE = 44100  # Sampling rate
DURATION = 0.1  # Tone playback duration (seconds)


def generate_tone(frequency, duration, volume):
    """
    Generate a fixed stereo tone (same for left and right).
    volume: range from 0 to 1
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    # Stereo: same for left and right
    stereo = np.column_stack((tone, tone))
    return stereo.astype(np.float32).tobytes()


def map_depth_to_volume(depth_value, max_distance=10.0, exponent=2):
    """
    Function to apply non-linear (squared here) mapping so that the smaller the depth_value, the louder the volume.
    The larger the exponent value, the more emphasized the volume change at close range.
    """
    # Normalize to range 0 to 1
    ratio = max(0.0, min(1.0, (max_distance - depth_value) / max_distance))
    # Non-linear mapping (e.g., squared)
    volume = ratio**exponent
    return volume


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

    # Initialize audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=SAMPLE_RATE, output=True)

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
        volume = map_depth_to_volume(depth_center, max_distance=2.0, exponent=1)

        # Generate a tone with a fixed frequency (e.g., 440Hz) (play the same for left and right)
        tone = generate_tone(440, DURATION, volume)
        stream.write(tone)

        display.show(overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        lidar.event.clear()

    # Cleanup
    display.close()
    pc_processor.vis.destroy_window()
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main(dev_idx=0)
