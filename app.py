"""Worker thử đồ 2D.

Chạy: HF_TOKEN=... python -m uvicorn app:app --host 127.0.0.1 --port 8095

POST /check-person  file ảnh -> {"hasPerson": true|false}
    Có HF_TOKEN thì gọi DETR trên Hugging Face.
    Chưa có token thì dùng YOLOv8n chạy local.
POST /try-on        person + garment -> ảnh PNG

Space mặc định là bản CatVTON đang chạy (cùng model SD 1.5).
Space gốc zhengchong/CatVTON hiện không khởi động được vì model FLUX bị khóa.
Đổi space bằng biến TRYON_SPACE.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from gradio_client import Client, handle_file
from PIL import Image

log = logging.getLogger("tryon-worker")
logging.basicConfig(level=logging.INFO)


def _load_local_env() -> None:
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()

MAX_BYTES = 8 * 1024 * 1024
PERSON_SCORE = 0.5
DETECTOR_MODEL = os.getenv("HF_DETECTOR_MODEL", "facebook/detr-resnet-50")
TRYON_SPACE = os.getenv("TRYON_SPACE", "multimodalart/CatVTON-zerogpu")
TRYON_API = os.getenv("TRYON_API_NAME", "/submit_function")
CLOTH_TYPES = {"upper", "lower", "overall"}

app = FastAPI(title="tryon-worker")
_client: Client | None = None
_client_lock = threading.Lock()


def _token() -> str | None:
    token = os.getenv("HF_TOKEN", "").strip()
    return token or None


def _open_image(raw: bytes) -> Image.Image:
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Ảnh vượt quá 8MB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ") from exc
    return image.convert("RGB")


def _limit_side(image: Image.Image, max_side: int = 1024) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _save_temp(image: Image.Image, suffix: str = ".jpg") -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = handle.name
    handle.close()
    image.save(path, format="PNG" if suffix == ".png" else "JPEG", quality=90)
    return path


def _blank_mask(width: int, height: int) -> str:
    # Mask một màu để Space tự tách vùng áo. Để đuôi .png vì Gradio lưu ảnh editor
    # ở chế độ RGBA và không ghi được JPEG.
    mask = Image.new("RGB", (max(1, width), max(1, height)), (0, 0, 0))
    return _save_temp(mask, ".png")


async def _read_upload(upload: UploadFile) -> Image.Image:
    raw = await upload.read()
    return _open_image(raw)


def _detections(image_bytes: bytes) -> list[dict]:
    headers = {"Content-Type": "image/jpeg"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://router.huggingface.co/hf-inference/models/{DETECTOR_MODEL}"

    def post() -> httpx.Response:
        with httpx.Client(timeout=40) as http:
            return http.post(url, content=image_bytes, headers=headers)

    response = post()
    if response.status_code == 503:
        wait = 8
        try:
            wait = min(float(response.json().get("estimated_time", 8)), 20)
        except Exception:
            wait = 8
        import time

        time.sleep(wait)
        response = post()

    if response.status_code >= 400:
        log.warning("Person detector failed: %s %s", response.status_code, response.text[:300])
        raise HTTPException(
            status_code=502,
            detail="Không kiểm tra được ảnh người. Hãy đặt HF_TOKEN rồi thử lại.",
        )

    payload = response.json()
    if isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Bộ phát hiện người không trả danh sách nhãn.")
    if not isinstance(payload, list):
        raise HTTPException(status_code=502, detail="Bộ phát hiện người trả dữ liệu lạ.")
    return payload


def _has_person(detections: list[dict]) -> bool:
    for item in detections:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0))
        if "person" in label and score >= PERSON_SCORE:
            return True
    return False


YOLO_URL = os.getenv(
    "YOLO_URL",
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
)
_yolo_session = None


def _ensure_yolo() -> Path:
    cache = Path(__file__).resolve().parent / ".cache"
    cache.mkdir(exist_ok=True)
    path = cache / "yolov8n.onnx"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    log.info("Downloading YOLOv8n for local person detection")
    with httpx.Client(timeout=120, follow_redirects=True) as http:
        response = http.get(YOLO_URL)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def _yolo():
    global _yolo_session
    if _yolo_session is None:
        import onnxruntime as ort

        _yolo_session = ort.InferenceSession(str(_ensure_yolo()), providers=["CPUExecutionProvider"])
    return _yolo_session


def _has_person_local(image: Image.Image) -> bool:
    """YOLOv8n local khi chưa có HF_TOKEN. DETR trên Hugging Face cần token."""
    size = 640
    fitted = image.copy()
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    blob = np.transpose(np.array(canvas).astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    session = _yolo()
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: blob})[0]
    # (1, 84, N) or (1, N, 84). Class 0 is "person", sitting just after the 4 box values.
    if prediction.shape[1] == 84:
        person_scores = prediction[0, 4, :]
    elif prediction.shape[-1] == 84:
        person_scores = prediction[0, :, 4]
    else:
        raise HTTPException(status_code=502, detail="Model phát hiện người trả về kích thước lạ.")
    return float(np.max(person_scores)) >= PERSON_SCORE


def _space_client() -> Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = Client(TRYON_SPACE, token=_token())
        return _client


def _drop_client() -> None:
    global _client
    with _client_lock:
        _client = None


def _png_bytes(result) -> bytes:
    if isinstance(result, (list, tuple)):
        if not result:
            raise HTTPException(status_code=502, detail="Space không trả ảnh.")
        result = result[0]
    if isinstance(result, dict):
        result = result.get("path") or result.get("url") or result.get("name")
    if not isinstance(result, str) or not result:
        raise HTTPException(status_code=502, detail="Space trả về định dạng ảnh không đọc được.")

    if result.startswith("http://") or result.startswith("https://"):
        with httpx.Client(timeout=60) as http:
            downloaded = http.get(result)
            downloaded.raise_for_status()
            raw = downloaded.content
    else:
        raw = Path(result).read_bytes()

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _call_space(person_path: str, mask_path: str, garment_path: str, cloth_type: str):
    person_editor = {
        "background": handle_file(person_path),
        "layers": [handle_file(mask_path)],
        "composite": None,
        "id": None,
    }
    client = _space_client()
    return client.predict(
        person_image=person_editor,
        cloth_image=handle_file(garment_path),
        cloth_type=cloth_type,
        num_inference_steps=30,
        guidance_scale=2.5,
        seed=42,
        show_type="result only",
        api_name=TRYON_API,
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "space": TRYON_SPACE,
        "api": TRYON_API,
        "tokenConfigured": _token() is not None,
        "personDetector": "detr" if _token() else "yolov8n",
    }


@app.post("/check-person")
async def check_person(person: UploadFile = File(...)):
    image = _limit_side(await _read_upload(person))
    if _token():
        detections = _detections(_jpeg_bytes(image))
        found = _has_person(detections)
    else:
        found = _has_person_local(image)
    return {"hasPerson": found}


@app.post("/try-on")
async def try_on(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    cloth_type: str = Form("upper"),
):
    cloth_type = (cloth_type or "upper").strip().lower()
    if cloth_type not in CLOTH_TYPES:
        raise HTTPException(status_code=400, detail="cloth_type phải là upper, lower hoặc overall.")

    person_image = _limit_side(await _read_upload(person))
    garment_image = _limit_side(await _read_upload(garment))
    paths: list[str] = []
    try:
        person_path = _save_temp(person_image, ".png")
        garment_path = _save_temp(garment_image, ".png")
        mask_path = _blank_mask(*person_image.size)
        paths.extend([person_path, garment_path, mask_path])
        try:
            result = _call_space(person_path, mask_path, garment_path, cloth_type)
        except Exception:
            _drop_client()
            log.exception("CatVTON call failed")
            raise HTTPException(
                status_code=502,
                detail="Space CatVTON không trả ảnh. Space có thể đang xếp hàng hoặc vừa ngủ.",
            ) from None
        return Response(content=_png_bytes(result), media_type="image/png")
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass
