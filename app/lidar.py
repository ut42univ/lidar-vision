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
        self.vis = self._initialize_visualizer()

    def _initialize_visualizer(self) -> o3d.visualization.Visualizer:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud")
        return vis

    def on_new_frame(self) -> None:
        """非メインスレッドから呼び出され、新フレーム到着を通知する"""
        self.event.set()

    def on_stream_stopped(self) -> None:
        print("Stream stopped")

    def connect_to_device(self, dev_idx: int) -> None:
        print("Searching for devices")
        devices = Record3DStream.get_connected_devices()
        print(f"{len(devices)} device(s) found")
        for dev in devices:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}\n")

        if len(devices) <= dev_idx:
            raise RuntimeError(
                f"Cannot connect to device #{dev_idx}, try a different index."
            )

        selected_dev = devices[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(selected_dev)

    @staticmethod
    def get_intrinsic_mat_from_coeffs(coeffs) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def create_point_cloud(self) -> o3d.geometry.PointCloud:
        depth_frame = self.session.get_depth_frame()
        rgb_frame = self.session.get_rgb_frame()
        resized_rgb = cv2.resize(rgb_frame, np.flip(np.shape(depth_frame)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(resized_rgb),
            o3d.geometry.Image(depth_frame),
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

    def save_point_cloud(
        self, pcd: o3d.geometry.PointCloud, filename: str = "captured_pointcloud.ply"
    ) -> None:
        o3d.io.write_point_cloud(filename, pcd)

    def record_frame(self) -> None:
        if self.recording:
            self.recorded_frames.append(copy.deepcopy(self.pcd))

    def save_recording(self, filename: str = "recording.ply") -> None:
        if self.recorded_frames:
            combined = self.recorded_frames[0]
            for frame in self.recorded_frames[1:]:
                combined += frame
            self.save_point_cloud(combined, filename)
            self.recorded_frames = []

    def process_point_cloud(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        if self.filter_enabled:
            # 統計的外れ値除去
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # ボクセルダウンサンプリング
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            # 法線推定
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        return pcd

    def _setup_intrinsics(self) -> None:
        depth_frame = self.session.get_depth_frame()
        coeffs = self.session.get_intrinsic_mat()
        intrinsic_mat = self.get_intrinsic_mat_from_coeffs(coeffs)
        height, width = np.shape(depth_frame)
        # 深度マップのサイズに合わせて内部パラメータを1/4にスケールダウン
        fx = intrinsic_mat[0, 0] / 4
        fy = intrinsic_mat[1, 1] / 4
        cx = intrinsic_mat[0, 2] / 4
        cy = intrinsic_mat[1, 2] / 4
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

    def _update_visualization(self, updated_pcd: o3d.geometry.PointCloud) -> None:
        self.pcd.points = updated_pcd.points
        self.pcd.colors = updated_pcd.colors
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def start_processing_stream(self) -> None:
        # 最初のフレーム待機
        self.event.wait()
        self._setup_intrinsics()
        self.pcd = self.create_point_cloud()
        self.vis.add_geometry(self.pcd)
        # RGB映像表示用のウィンドウを作成
        cv2.namedWindow("RGB Video", cv2.WINDOW_NORMAL)

        while True:
            self.event.wait()
            # 最新のRGBフレームと深度フレームを取得
            rgb_frame = self.session.get_rgb_frame()
            depth_frame = self.session.get_depth_frame()

            # RGB画像をBGRに変換（OpenCVはBGRを期待）
            rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 深度画像の中心ピクセルの奥行き値を取得
            center_y = depth_frame.shape[0] // 2
            center_x = depth_frame.shape[1] // 2
            depth_center = depth_frame[center_y, center_x]

            # 深度値をテキストとして画像に描画
            text = f"Depth at Centre: {depth_center}"
            cv2.putText(
                rgb_bgr,
                text,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 中心に十字（クロスマーカー）を描画
            cv2.drawMarker(
                rgb_bgr,
                (rgb_bgr.shape[1] // 2, rgb_bgr.shape[0] // 2),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=40,
                thickness=4,
            )

            # ポイントクラウドの更新処理
            new_pcd = self.create_point_cloud()
            processed_pcd = self.process_point_cloud(new_pcd)
            self.record_frame()
            self._update_visualization(processed_pcd)

            # RGB映像の表示
            cv2.imshow("RGB Video", rgb_bgr)
            # 'q'キーでループ終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            self.event.clear()

        # 終了処理
        cv2.destroyAllWindows()
        self.vis.destroy_window()

    def run(self, dev_idx: int = 0) -> None:
        self.connect_to_device(dev_idx)
        self.start_processing_stream()


if __name__ == "__main__":
    app = LidarApp()
    app.run(dev_idx=0)
