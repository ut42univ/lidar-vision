# main.py
import cv2
from cv2display import CV2Display
from lidar import LidarDevice
from pointcloud import PointCloudProcessor


def main(dev_idx: int = 0) -> None:
    # Initialize and connect to the LiDAR device
    lidar = LidarDevice()
    lidar.connect(dev_idx)

    # Initialize point cloud processing
    pc_processor = PointCloudProcessor()

    # Initialize cv2 display
    display = CV2Display()

    # Wait for the first frame and set up intrinsic parameters
    lidar.event.wait()
    pc_processor.setup_intrinsics(lidar.session)
    pc_processor.pcd = pc_processor.create_point_cloud(lidar.session)
    pc_processor.vis.add_geometry(pc_processor.pcd)

    while True:
        lidar.event.wait()
        rgb_frame = lidar.session.get_rgb_frame()
        depth_frame = lidar.session.get_depth_frame()

        # Draw overlay with cv2
        overlay = display.draw_overlay(rgb_frame, depth_frame)

        # Generate, process, record, and update point cloud
        new_pcd = pc_processor.create_point_cloud(lidar.session)
        processed_pcd = pc_processor.process_point_cloud(new_pcd)
        pc_processor.record_frame()
        pc_processor.update_visualization(processed_pcd)

        display.show(overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        lidar.event.clear()

    display.close()
    pc_processor.vis.destroy_window()


if __name__ == "__main__":
    main(dev_idx=0)
