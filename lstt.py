#!/usr/bin/env python3
"""lstt - Push-to-talk speech transcription for Linux/Wayland."""

import subprocess
import sys
import threading
from pathlib import Path

import evdev


def notify(title: str, message: str = "", urgency: str = "normal"):
    """Send a desktop notification."""
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, "-a", "lstt", title, message],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Notification failed, continue silently
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Configuration
WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000
CHANNELS = 1

# Key codes
KEY_LEFTCTRL = 29
KEY_LEFTMETA = 125  # Super/Windows key


class AudioRecorder:
    """Records audio from microphone to numpy array."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream = None

    def start(self):
        """Start recording audio."""
        self.audio_data = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.audio_data).flatten()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if self.recording:
            self.audio_data.append(indata.copy())


class Transcriber:
    """Transcribes audio using faster-whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        print(f"Loading Whisper model '{model_name}'...")
        notify("lstt", f"Loading model '{model_name}'... (may download ~3GB)")
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print("Model loaded.")
        notify("lstt", "Model loaded. Ready!")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array to text."""
        if len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(audio, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        return text.strip()


class TextTyper:
    """Types text using ydotool."""

    @staticmethod
    def type_text(text: str):
        """Type text at current cursor position."""
        if not text:
            return

        try:
            subprocess.run(
                ["ydotool", "type", "--", text],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"ydotool error: {e.stderr.decode()}")
        except FileNotFoundError:
            print("ydotool not found. Install with: sudo apt install ydotool")


class HotkeyMonitor:
    """Monitors for Ctrl+Super hotkey using evdev."""

    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.ctrl_pressed = False
        self.meta_pressed = False
        self.combo_active = False
        self.devices = []

    def find_keyboards(self) -> list[evdev.InputDevice]:
        """Find all keyboard input devices."""
        keyboards = []
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                caps = device.capabilities()
                # Check if device has EV_KEY capability with keyboard keys
                if evdev.ecodes.EV_KEY in caps:
                    key_caps = caps[evdev.ecodes.EV_KEY]
                    if KEY_LEFTCTRL in key_caps and KEY_LEFTMETA in key_caps:
                        keyboards.append(device)
            except (PermissionError, OSError):
                continue
        return keyboards

    def run(self):
        """Start monitoring for hotkey."""
        self.devices = self.find_keyboards()

        if not self.devices:
            print("No keyboard devices found. Make sure you're in the 'input' group.")
            print("Run: sudo usermod -aG input $USER")
            sys.exit(1)

        print(f"Monitoring {len(self.devices)} keyboard device(s)")
        print("Hold Ctrl+Super to record, release to transcribe.")
        print("Press Ctrl+C to exit.\n")

        # Monitor all keyboards in separate threads
        threads = []
        for device in self.devices:
            t = threading.Thread(target=self._monitor_device, args=(device,), daemon=True)
            t.start()
            threads.append(t)

        # Keep main thread alive
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nExiting...")

    def _monitor_device(self, device: evdev.InputDevice):
        """Monitor a single input device."""
        try:
            for event in device.read_loop():
                if event.type != evdev.ecodes.EV_KEY:
                    continue

                key_event = evdev.categorize(event)

                if event.code == KEY_LEFTCTRL:
                    self.ctrl_pressed = key_event.keystate != 0
                elif event.code == KEY_LEFTMETA:
                    self.meta_pressed = key_event.keystate != 0

                # Check combo state
                combo_now = self.ctrl_pressed and self.meta_pressed

                if combo_now and not self.combo_active:
                    self.combo_active = True
                    self.on_press()
                elif not combo_now and self.combo_active:
                    self.combo_active = False
                    self.on_release()

        except OSError:
            # Device disconnected
            pass


class Lstt:
    """Main application class."""

    def __init__(self):
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.typer = TextTyper()
        self.monitor = HotkeyMonitor(
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )

    def _on_hotkey_press(self):
        """Called when Ctrl+Super is pressed."""
        print("Recording...", end="", flush=True)
        notify("Recording...", "Release to transcribe")
        self.recorder.start()

    def _on_hotkey_release(self):
        """Called when Ctrl+Super is released."""
        audio = self.recorder.stop()
        duration = len(audio) / SAMPLE_RATE
        print(f" {duration:.1f}s captured.")

        if duration < 0.3:
            print("Recording too short, skipping.")
            notify("Recording too short", "Skipped")
            return

        print("Transcribing...", end="", flush=True)
        notify("Transcribing...", f"{duration:.1f}s of audio")
        text = self.transcriber.transcribe(audio)
        print(f" done.")

        if text:
            print(f"Text: {text}")
            notify("Transcribed", text)
            self.typer.type_text(text)
        else:
            print("No speech detected.")
            notify("No speech detected", "")

    def run(self):
        """Run the application."""
        self.monitor.run()


def main():
    app = Lstt()
    app.run()


if __name__ == "__main__":
    main()
