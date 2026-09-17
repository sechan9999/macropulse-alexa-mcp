"""
src/ambient_audio.py
─────────────────────────────────────────────────────────────────
Personalized Ambient Audio Chime Engine for Macro Pulse & Fire TV.

Provides:
1. Procedural WAV audio generation using Python standard library (wave, math, struct)
   encoded as base64 data URIs for instant zero-dependency playback.
2. Web Audio API JavaScript synthesizers for real-time, low-latency ambient cues
   on Fire TV and browser clients.
3. Event triggers:
   - High-probability Bollinger Squeeze Breakout (Bullish / Bearish)
   - Corporate Credit Spread Divergence (BAA-AAA widening vs S&P highs)
   - NVDA Danger Zone alert (Composite Danger Index > 0.70)
   - FOMC Rate Decision Shock Alert
"""
from __future__ import annotations

import base64
import io
import math
import struct
from typing import Dict, Literal, Optional

ChimeType = Literal["bollinger_breakout", "credit_divergence", "danger_zone", "fomc_shock"]


def generate_chime_wav(
    chime_type: ChimeType = "bollinger_breakout",
    duration_sec: float = 1.6,
    sample_rate: int = 22050,
    volume: float = 0.65
) -> bytes:
    """
    Generates a high-quality ambient harmonic chime as raw WAV bytes.
    Uses multi-frequency additive synthesis with exponential decay envelope.
    """
    num_samples = int(sample_rate * duration_sec)

    # Define musical chord frequencies (Hz) and relative overtone weights
    if chime_type == "bollinger_breakout":
        # Bright, crisp ascending marimba / celestial chord: C5, E5, G5, C6
        notes = [
            (523.25, 0.40, 0.0),    # C5, start 0.0s
            (659.25, 0.35, 0.08),   # E5, start 0.08s
            (783.99, 0.30, 0.16),   # G5, start 0.16s
            (1046.50, 0.45, 0.24),  # C6, start 0.24s
        ]
        decay_rate = 3.2
    elif chime_type == "credit_divergence":
        # Deep warm marimba / low gong chord: C3, Eb3, G3, Bb3 (Minor 7th)
        notes = [
            (130.81, 0.50, 0.0),    # C3
            (155.56, 0.40, 0.04),   # Eb3
            (196.00, 0.35, 0.08),   # G3
            (233.08, 0.25, 0.12),   # Bb3
        ]
        decay_rate = 2.0
    elif chime_type == "danger_zone":
        # Pulsed minor second / tritone caution pulse: F#4, C5, F5
        notes = [
            (369.99, 0.45, 0.0),    # F#4
            (523.25, 0.40, 0.05),   # C5 (tritone)
            (698.46, 0.30, 0.10),   # F5
        ]
        decay_rate = 3.8
    elif chime_type == "fomc_shock":
        # Deep resonant bell: A3, E4, A4, C#5
        notes = [
            (220.00, 0.50, 0.0),    # A3
            (329.63, 0.40, 0.06),   # E4
            (440.00, 0.35, 0.12),   # A4
            (554.37, 0.30, 0.18),   # C#5
        ]
        decay_rate = 2.4
    else:
        notes = [(440.0, 0.5, 0.0)]
        decay_rate = 3.0

    raw_samples = [0.0] * num_samples

    for freq, note_vol, start_sec in notes:
        start_idx = int(start_sec * sample_rate)
        if start_idx >= num_samples:
            continue
        
        note_len = num_samples - start_idx
        for i in range(note_len):
            t = i / sample_rate
            # Smooth attack envelope (15ms) + exponential release decay
            attack = min(1.0, t / 0.015)
            envelope = attack * math.exp(-decay_rate * t)
            
            # Fundamental + soft second harmonic
            val = (
                math.sin(2.0 * math.pi * freq * t) +
                0.35 * math.sin(2.0 * math.pi * freq * 2.0 * t)
            ) * note_vol * envelope
            
            raw_samples[start_idx + i] += val

    # Normalize and convert to 16-bit signed PCM
    max_val = max((abs(s) for s in raw_samples), default=1.0)
    norm_factor = (32767.0 * volume) / max(max_val, 1e-6)

    pcm_data = bytearray()
    for s in raw_samples:
        sample_int = int(max(-32768, min(32767, s * norm_factor)))
        pcm_data.extend(struct.pack("<h", sample_int))

    # Construct WAV file
    buf = io.BytesIO()
    with io.BytesIO() as wav_file:
        # Header
        num_channels = 1
        bytes_per_sample = 2
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        data_size = len(pcm_data)
        chunk_size = 36 + data_size

        wav_file.write(b"RIFF")
        wav_file.write(struct.pack("<I", chunk_size))
        wav_file.write(b"WAVE")
        wav_file.write(b"fmt ")
        wav_file.write(struct.pack("<I", 16))          # Subchunk1Size (16 for PCM)
        wav_file.write(struct.pack("<H", 1))           # AudioFormat (1 for PCM)
        wav_file.write(struct.pack("<H", num_channels))
        wav_file.write(struct.pack("<I", sample_rate))
        wav_file.write(struct.pack("<I", byte_rate))
        wav_file.write(struct.pack("<H", block_align))
        wav_file.write(struct.pack("<H", bytes_per_sample * 8)) # BitsPerSample
        wav_file.write(b"data")
        wav_file.write(struct.pack("<I", data_size))
        wav_file.write(pcm_data)
        return wav_file.getvalue()


def get_chime_base64_data_uri(chime_type: ChimeType = "bollinger_breakout", volume: float = 0.65) -> str:
    """Returns a base64-encoded data URI string: 'data:audio/wav;base64,...'"""
    wav_bytes = generate_chime_wav(chime_type=chime_type, volume=volume)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def get_web_audio_inline_js(chime_type: ChimeType = "bollinger_breakout", volume: float = 0.5) -> str:
    """
    Returns inline JavaScript that uses the Web Audio API to procedurally play
    the specified ambient chime on any browser or Fire TV webview.
    """
    freqs_map = {
        "bollinger_breakout": "[523.25, 659.25, 783.99, 1046.50]",
        "credit_divergence": "[130.81, 155.56, 196.00, 233.08]",
        "danger_zone": "[369.99, 523.25, 698.46]",
        "fomc_shock": "[220.00, 329.63, 440.00, 554.37]"
    }
    freqs = freqs_map.get(chime_type, "[440, 660, 880]")
    
    return f"""
    (function() {{
      try {{
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        if (ctx.state === 'suspended') {{
          ctx.resume();
        }}
        const freqs = {freqs};
        const baseVol = {volume};
        freqs.forEach((freq, idx) => {{
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, ctx.currentTime);
          const startT = ctx.currentTime + (idx * 0.08);
          gain.gain.setValueAtTime(0.001, startT);
          gain.gain.linearRampToValueAtTime(baseVol * 0.4, startT + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, startT + 1.4);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(startT);
          osc.stop(startT + 1.5);
        }});
      }} catch(e) {{
        console.warn("Web Audio ambient chime error:", e);
      }}
    }})();
    """
