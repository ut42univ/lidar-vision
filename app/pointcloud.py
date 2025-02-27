# pointcloud.py
import copy

import cv2
import numpy as np
import open3d as o3d


class PointCloudProcessor:
    def __init__(self):
        self.intrinsic = None
        self.pcd = None
        self.recorded_frames = []
        self.filter_enabled = False
        self.vis = self._init_visualizer()

    def _init_visualizer(self) -> o3d.visualization.Visualizer:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud")
        return vis

    @staticmethod
    def _get_intrinsic_matrix(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def setup_intrinsics(self, session) -> None:
        depth_frame = session.get_depth_frame()
        coeffs = session.get_intrinsic_mat()
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

    def create_point_cloud(self, session) -> o3d.geometry.PointCloud:
        depth_frame = session.get_depth_frame()
        rgb_frame = session.get_rgb_frame()
        # Resize RGB image to match the size of the depth frame
        resized_rgb = cv2.resize(rgb_frame, tuple(np.flip(depth_frame.shape)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(resized_rgb),
            o3d.geometry.Image(depth_frame),
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

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

    def update_visualization(self, updated_pcd: o3d.geometry.PointCloud) -> None:
        self.pcd.points = updated_pcd.points
        self.pcd.colors = updated_pcd.colors
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def record_frame(self) -> None:
        if self.pcd is not None:
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
