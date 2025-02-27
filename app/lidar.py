# lidar.py
from threading import Event

from record3d import Record3DStream  # type: ignore


class LidarDevice:
    def __init__(self):
        self.event = Event()
        self.session = None

    def on_new_frame(self) -> None:
        """Event notification when a frame is received (called from a non-main thread)"""
        self.event.set()

    def on_stream_stopped(self) -> None:
        print("Stream stopped")

    def connect(self, dev_idx: int) -> None:
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
