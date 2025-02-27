import copy
from threading import Event

import cv2
import numpy as np
import open3d as o3d
from record3d import Record3DStream  # type: ignore


class LidarApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.filter_enabled = False
        self.recording = False
        self.recorded_frames = []
        self.intrinsic = None
        self.pcd = None
        self.vis = self._init_visualizer()

    def _init_visualizer(self) -> o3d.visualization.Visualizer:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud")
        return vis

    def on_new_frame(self) -> None:
        """Called from non-main thread to notify arrival of a new frame"""
        self.event.set()

    def on_stream_stopped(self) -> None:
        print("Stream stopped")

    def connect_to_device(self, dev_idx: int) -> None:
        print("Searching for devices")
        devices = Record3DStream.get_connected_devices()
        print(f"{len(devices)} device(s) found")
        for dev in devices:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}\n")

        if dev_idx >= len(devices):
            raise RuntimeError(
                f"Cannot connect to device #{dev_idx}, try a different index."
            )

        selected_dev = devices[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(selected_dev)

    @staticmethod
    def _get_intrinsic_matrix(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def _setup_intrinsics(self) -> None:
        depth_frame = self.session.get_depth_frame()
        coeffs = self.session.get_intrinsic_mat()
        intrinsic_mat = self._get_intrinsic_matrix(coeffs)
        height, width = depth_frame.shape
        # Scale down intrinsic parameters by 1/4 to match the size of the depth map
        fx = intrinsic_mat[0, 0] / 4
        fy = intrinsic_mat[1, 1] / 4
        cx = intrinsic_mat[0, 2] / 4
        cy = intrinsic_mat[1, 2] / 4
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

    def create_point_cloud(self) -> o3d.geometry.PointCloud:
        depth_frame = self.session.get_depth_frame()
        rgb_frame = self.session.get_rgb_frame()
        resized_rgb = cv2.resize(rgb_frame, tuple(np.flip(depth_frame.shape)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(resized_rgb),
            o3d.geometry.Image(depth_frame),
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

    def _update_visualization(self, updated_pcd: o3d.geometry.PointCloud) -> None:
        self.pcd.points = updated_pcd.points
        self.pcd.colors = updated_pcd.colors
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def process_point_cloud(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        if self.filter_enabled:
            # Statistical outlier removal
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # Voxel downsampling
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            # Normal estimation
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        return pcd

    def record_frame(self) -> None:
        if self.recording:
            self.recorded_frames.append(copy.deepcopy(self.pcd))

    def save_point_cloud(
        self, pcd: o3d.geometry.PointCloud, filename: str = "captured_pointcloud.ply"
    ) -> None:
        o3d.io.write_point_cloud(filename, pcd)

    def save_recording(self, filename: str = "recording.ply") -> None:
        if not self.recorded_frames:
            return
        combined = self.recorded_frames[0]
        for frame in self.recorded_frames[1:]:
            combined += frame
        self.save_point_cloud(combined, filename)
        self.recorded_frames = []

    def _draw_rgb_overlay(
        self, rgb_frame: np.ndarray, depth_frame: np.ndarray
    ) -> np.ndarray:
        """Draw depth information and a central cross marker on the RGB image"""
        rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        center_y, center_x = depth_frame.shape[0] // 2, depth_frame.shape[1] // 2
        depth_center = depth_frame[center_y, center_x]
        text = f"Depth at Centre: {depth_center}"
        cv2.putText(
            rgb_bgr, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3
        )
        cv2.drawMarker(
            rgb_bgr,
            (rgb_bgr.shape[1] // 2, rgb_bgr.shape[0] // 2),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=4,
        )
        return rgb_bgr

    def start_processing_stream(self) -> None:
        # Wait for the first frame and set up intrinsic parameters
        self.event.wait()
        self._setup_intrinsics()
        self.pcd = self.create_point_cloud()
        self.vis.add_geometry(self.pcd)
        cv2.namedWindow("RGB Video", cv2.WINDOW_AUTOSIZE)

        while True:
            self.event.wait()
            rgb_frame = self.session.get_rgb_frame()
            depth_frame = self.session.get_depth_frame()

            # Draw overlay on RGB video
            overlay = self._draw_rgb_overlay(rgb_frame, depth_frame)

            # Generate, process, and update point cloud
            new_pcd = self.create_point_cloud()
            processed_pcd = self.process_point_cloud(new_pcd)
            self.record_frame()
            self._update_visualization(processed_pcd)

            cv2.imshow("RGB Video", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            self.event.clear()

        cv2.destroyAllWindows()
        self.vis.destroy_window()

    def run(self, dev_idx: int = 0) -> None:
        self.connect_to_device(dev_idx)
        self.start_processing_stream()


if __name__ == "__main__":
    app = LidarApp()
    app.run(dev_idx=0)
