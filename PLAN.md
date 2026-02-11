# Plan: Native feel + confidence detection

All changes in `lstt.py` only. No new files, no new dependencies.

## 1. Extract confidence scores from faster-whisper

Change `Transcriber.transcribe()` to return confidence info alongside text:
- Collect `avg_logprob` and `no_speech_prob` from each segment
- Return a result object/tuple with: text, avg confidence, max no-speech probability

## 2. Low-confidence notification

In `_on_hotkey_release()`, after transcription:
- If `no_speech_prob` is high (>0.6) or `avg_logprob` is very low (<-1.0): notify with "Low confidence" warning but **still type the text**
- If no text at all: keep current "No speech detected" behavior

## 3. In-memory history (last 8)

- Add a `collections.deque(maxlen=8)` to the `Lstt` class
- After each transcription, append an entry: `{timestamp, text, confidence, duration}`
- Print the history buffer to console on each new transcription (compact format)
- That's it — no persistence, no GUI, just in-memory with console visibility

## 4. Clean up imports

- Move the misplaced imports (numpy, sounddevice, faster_whisper) to the top with the rest
- Remove unused `from pathlib import Path`
