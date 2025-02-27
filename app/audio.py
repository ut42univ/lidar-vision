import numpy as np
import pyaudio

SAMPLE_RATE = 44100  # Sampling rate
DURATION = 0.1  # Tone playback duration (seconds)


class AudioGenerator:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32, channels=2, rate=SAMPLE_RATE, output=True
        )

    def generate_tone(self, frequency, duration, volume):
        """
        Generate a fixed stereo tone (same for left and right).
        volume: range from 0 to 1
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t) * volume
        # Stereo: same for left and right
        stereo = np.column_stack((tone, tone))
        return stereo.astype(np.float32).tobytes()

    def map_depth_to_volume(self, depth_value, max_distance=10.0, exponent=2):
        """
        Function to apply non-linear (squared here) mapping so that the smaller the depth_value, the louder the volume.
        The larger the exponent value, the more emphasized the volume change at close range.
        """
        # Normalize to range 0 to 1
        ratio = max(0.0, min(1.0, (max_distance - depth_value) / max_distance))
        # Non-linear mapping (e.g., squared)
        volume = ratio**exponent
        return volume

    def play_tone(self, frequency, volume):
        tone = self.generate_tone(frequency, DURATION, volume)
        self.stream.write(tone)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
