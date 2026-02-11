# lstt

Push-to-talk speech transcription tool for Linux/Wayland.

## How it works
- Hold Ctrl+Super to record audio
- Release to transcribe and type at cursor position
- Uses faster-whisper for local transcription (no cloud API)

## Setup (Ubuntu 24)

### System dependencies
```bash
sudo apt install ydotool libportaudio2
```

### ydotool daemon
```bash
sudo systemctl enable --now ydotool
```

### Input group (for keyboard access)
```bash
sudo usermod -aG input $USER
```
Log out and back in for the group change to take effect.

### Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python toritalk.py
```

Hold Ctrl+Super, speak, and release to transcribe.

## Configuration

Edit constants in `toritalk.py`:

### Whisper model
`WHISPER_MODEL` - Model size (downloads automatically on first run):
- `tiny.en` - fastest, least accurate (~75MB)
- `base.en` - good balance (~150MB) **default**
- `small.en` - better accuracy (~500MB)
- `medium.en` - high accuracy (~1.5GB)
- `large-v3` - best accuracy (~3GB)

### Audio
- `SAMPLE_RATE` - Audio sample rate (16000 for Whisper)
