#!/usr/bin/env python3
"""lstt - Push-to-talk speech transcription for Linux/Wayland."""

import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import evdev
import gi
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gdk, GLib, Gtk, GtkLayerShell


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

# Configuration
WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000
CHANNELS = 1
LOW_CONFIDENCE_LOGPROB = -1.0
HIGH_NO_SPEECH_PROB = 0.6


@dataclass
class TranscriptionResult:
    text: str
    avg_logprob: float
    no_speech_prob: float
    duration: float
    timestamp: str


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

    def transcribe(self, audio: np.ndarray, duration: float, on_segment=None) -> TranscriptionResult:
        """Transcribe audio array to text with confidence info."""
        if len(audio) == 0:
            return TranscriptionResult("", 0.0, 1.0, duration, datetime.now().strftime("%H:%M:%S"))

        segments, _ = self.model.transcribe(audio, beam_size=5)
        texts = []
        logprobs = []
        no_speech_probs = []
        for segment in segments:
            texts.append(segment.text)
            logprobs.append(segment.avg_logprob)
            no_speech_probs.append(segment.no_speech_prob)
            if on_segment:
                on_segment(" ".join(texts).strip())

        text = " ".join(texts).strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
        max_no_speech = max(no_speech_probs) if no_speech_probs else 1.0

        return TranscriptionResult(text, avg_logprob, max_no_speech, duration, datetime.now().strftime("%H:%M:%S"))


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


class AudioDucker:
    """Ducks system audio volume during recording via PipeWire/wpctl."""

    DUCK_VOLUME = "20%"

    def __init__(self):
        self.original_volume = None

    def duck(self):
        try:
            result = subprocess.run(
                ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                capture_output=True, text=True,
            )
            # Output like "Volume: 0.50" or "Volume: 0.50 [MUTED]"
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                self.original_volume = parts[1]
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", self.DUCK_VOLUME],
                capture_output=True,
            )
        except FileNotFoundError:
            pass

    def restore(self):
        if self.original_volume is None:
            return
        try:
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", self.original_volume],
                capture_output=True,
            )
        except FileNotFoundError:
            pass
        self.original_volume = None


class RecordingIndicator:
    """Floating overlay indicator using GTK Layer Shell."""

    CSS = """
    .indicator {
        background-color: rgba(30, 30, 30, 0.9);
        border-radius: 16px;
        padding: 6px 14px;
        color: white;
        font-size: 14px;
        font-weight: bold;
    }
    .indicator-recording {
        color: #ff4444;
    }
    .indicator-transcribing {
        color: #ffaa00;
    }
    """

    def __init__(self):
        # Load CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(self.CSS.encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        self.window = Gtk.Window()
        GtkLayerShell.init_for_window(self.window)
        GtkLayerShell.set_layer(self.window, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.RIGHT, True)
        GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.TOP, 24)
        GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.RIGHT, 24)
        GtkLayerShell.set_keyboard_mode(
            self.window, GtkLayerShell.KeyboardMode.NONE
        )

        self.label = Gtk.Label()
        self.label.get_style_context().add_class("indicator")
        self.window.add(self.label)
        self.window.show_all()
        self.window.hide()

    def show_recording(self):
        GLib.idle_add(self._show_recording)

    def _show_recording(self):
        self.label.set_markup(
            '<span color="#ff4444">●</span>  Recording...'
        )
        self.window.show_all()

    def show_transcribing(self, text=""):
        GLib.idle_add(self._show_transcribing, text)

    def _show_transcribing(self, text):
        display = GLib.markup_escape_text(text) if text else "..."
        self.label.set_markup(
            f'<span color="#ffaa00">⟳</span>  {display}'
        )
        self.window.show_all()

    def hide(self):
        GLib.idle_add(self.window.hide)


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
            GLib.idle_add(Gtk.main_quit)
            return

        print(f"Monitoring {len(self.devices)} keyboard device(s)")
        print("Hold Ctrl+Super to record, release to transcribe.")
        print("Press Ctrl+C to exit.\n")

        for device in self.devices:
            t = threading.Thread(target=self._monitor_device, args=(device,), daemon=True)
            t.start()

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
        self.indicator = RecordingIndicator()
        self.ducker = AudioDucker()
        self.history: deque[TranscriptionResult] = deque(maxlen=8)
        self.monitor = HotkeyMonitor(
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )

    def _on_hotkey_press(self):
        """Called when Ctrl+Super is pressed."""
        print("Recording...", end="", flush=True)
        self.ducker.duck()
        self.indicator.show_recording()
        self.recorder.start()

    def _on_hotkey_release(self):
        """Called when Ctrl+Super is released."""
        audio = self.recorder.stop()
        duration = len(audio) / SAMPLE_RATE
        print(f" {duration:.1f}s captured.")

        if duration < 0.3:
            print("Recording too short, skipping.")
            self.ducker.restore()
            self.indicator.hide()
            notify("Recording too short", "Skipped")
            return

        print("Transcribing...", end="", flush=True)
        self.indicator.show_transcribing()
        result = self.transcriber.transcribe(
            audio, duration, on_segment=self.indicator.show_transcribing
        )
        print(f" done.")
        self.ducker.restore()
        self.indicator.hide()

        if result.text:
            self.history.append(result)
            low_confidence = (
                result.avg_logprob < LOW_CONFIDENCE_LOGPROB
                or result.no_speech_prob > HIGH_NO_SPEECH_PROB
            )
            if low_confidence:
                print(f"Text (low confidence): {result.text}")
                notify("Low confidence", result.text, urgency="low")
            else:
                print(f"Text: {result.text}")
                notify("Transcribed", result.text)
            self.typer.type_text(result.text)
            self._print_history()
        else:
            print("No speech detected.")
            notify("No speech detected", "")

    def _print_history(self):
        """Print recent transcription history to console."""
        print(f"\n--- History ({len(self.history)}/8) ---")
        for r in self.history:
            confidence = "!" if r.avg_logprob < LOW_CONFIDENCE_LOGPROB or r.no_speech_prob > HIGH_NO_SPEECH_PROB else " "
            print(f"  [{r.timestamp}] {confidence} ({r.duration:.1f}s) {r.text}")
        print()

    def run(self):
        """Run the application."""
        # Hotkey monitor runs in background threads
        threading.Thread(target=self.monitor.run, daemon=True).start()
        # GTK main loop on the main thread
        Gtk.main()


def main():
    signal.signal(signal.SIGINT, lambda *_: Gtk.main_quit())
    app = Lstt()
    app.run()


if __name__ == "__main__":
    main()
