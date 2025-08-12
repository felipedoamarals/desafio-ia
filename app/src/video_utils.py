from pathlib import Path
from typing import List, Tuple
import cv2
import math
import numpy as np

def extract_frames(video_path: str, out_dir: str, interval_s: float = 0.1) -> List[Tuple[int, float, str]]:
    """Extrai frames a cada `interval_s` do vídeo.
    Salva PNGs temporários em `out_dir/frames` (útil para debug) e retorna lista [(frame_idx, ts_ms, png_path)].
    Em seguida, use image_utils.save_fits_to_db para persistir em FITS + Mongo.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    frames_dir = Path(out_dir) / "frames_tmp"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(interval_s * fps)))
    frame_idx = 0
    saved = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            png_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(png_path), frame)
            saved.append((frame_idx, float(ts_ms), str(png_path)))
        frame_idx += 1

    cap.release()
    return saved


def extract_audio_ffmpeg(video_path: str, out_wav: str, sr: int = 16000) -> str:
    """Extrai áudio para WAV mono usando ffmpeg (ffmpeg deve estar instalado na imagem)."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sr), str(out_wav)
    ]
    subprocess.run(cmd, check=True)
    return out_wav