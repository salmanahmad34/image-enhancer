"""
Local Image Enhancer Web Application
Powered by Gradio, Real-ESRGAN, and OpenCV.
Runs 100% locally with zero external paid APIs.
Supports BATCH processing of multiple images simultaneously.
"""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_SITE_PACKAGES = os.path.join(_CURRENT_DIR, "venv", "Lib", "site-packages")
if os.path.isdir(_VENV_SITE_PACKAGES) and _VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, _VENV_SITE_PACKAGES)

import re
import tempfile
import urllib.request
import logging
import socket
import contextlib
import io
import zipfile
import numpy as np  # type: ignore
import cv2  # type: ignore
import gradio as gr  # type: ignore

TORCH_AVAILABLE = False
try:
    with contextlib.redirect_stderr(io.StringIO()):
        import torch  # type: ignore
        TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    with contextlib.redirect_stderr(io.StringIO()):
        import torchvision.transforms.functional as F  # type: ignore
        import torchvision.transforms.functional_tensor  # type: ignore  # noqa
except ImportError:
    try:
        import torchvision.transforms.functional as F  # type: ignore
        import types
        functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
        functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
    except Exception:
        pass
except Exception:
    pass

REALESRGAN_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
            REALESRGAN_AVAILABLE = True
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_FILENAME = "RealESRGAN_x4plus.pth"
MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

_ESRGAN_UPSCALER = None


def download_model_weights(url: str, dest_path: str, progress_fn=None):
    """Download model weights with progress reporting if not already present."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        return dest_path
    logging.info(f"Downloading model weights from {url} to {dest_path}...")
    try:
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0 and progress_fn is not None:
                percent = min(1.0, (block_num * block_size) / total_size)
                progress_fn(percent, desc=f"Downloading AI weights... {int(percent * 100)}%")
        urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
        logging.info("Model weights download complete.")
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download model weights: {e}")
    return dest_path


def get_realesrgan_upscaler(progress_fn=None):
    """Lazy-load and cache the RealESRGANer model singleton."""
    global _ESRGAN_UPSCALER
    if _ESRGAN_UPSCALER is not None:
        return _ESRGAN_UPSCALER
    if not REALESRGAN_AVAILABLE:
        raise ImportError("Real-ESRGAN or BasicSR is not properly installed.")
    download_model_weights(MODEL_URL, MODEL_PATH, progress_fn=progress_fn)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Initializing RealESRGANer on device: {device}")
    _ESRGAN_UPSCALER = RealESRGANer(
        scale=4, model_path=MODEL_PATH, model=model,
        tile=400, tile_pad=10, pre_pad=0, half=use_cuda, device=device
    )
    return _ESRGAN_UPSCALER


# ---------------------------------------------------------------------------
# Enhancement Pipelines
# ---------------------------------------------------------------------------

def apply_unsharp_mask(image_bgr: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Apply single-pass unsharp mask filtering."""
    if amount <= 0:
        return image_bgr
    ksize = int(2 * round(radius * 3) + 1)
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), sigmaX=radius, sigmaY=radius)
    img_f = image_bgr.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    sharpened = img_f + amount * (img_f - blur_f)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_quick_opencv(
    image_bgr: np.ndarray,
    upscale_factor: float = 2.0,
    sharpness: float = 0.8,
    denoise_strength: float = 3.0,
    contrast: float = 1.5
) -> np.ndarray:
    """Quick enhancement pipeline using pure OpenCV algorithms."""
    processed = image_bgr.copy()
    if denoise_strength > 0:
        h = float(denoise_strength)
        processed = cv2.fastNlMeansDenoisingColored(
            processed, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
        )
    h_orig, w_orig = processed.shape[:2]
    new_w = max(1, int(round(w_orig * upscale_factor)))
    new_h = max(1, int(round(h_orig * upscale_factor)))
    if upscale_factor != 1.0 or (new_w != w_orig or new_h != h_orig):
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    if sharpness > 0:
        processed = apply_unsharp_mask(processed, radius=1.0, amount=sharpness)
        processed = apply_unsharp_mask(processed, radius=3.0, amount=sharpness * 0.5)
    if contrast > 0:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(contrast), tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return processed


def enhance_ai_realesrgan(
    image_bgr: np.ndarray,
    upscale_factor: float = 2.0,
    progress_fn=None
) -> np.ndarray:
    """AI Enhancement pipeline using Real-ESRGAN_x4plus."""
    h, w = image_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1200:
        scale_ratio = 1200.0 / max_dim
        new_w = int(round(w * scale_ratio))
        new_h = int(round(h * scale_ratio))
        logging.info(f"Input image ({w}x{h}) exceeds 1200px. Downscaling to ({new_w}x{new_h}).")
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    upscaler = get_realesrgan_upscaler(progress_fn=progress_fn)
    if progress_fn:
        progress_fn(0.3, desc="Running Real-ESRGAN AI enhancement...")
    output, _ = upscaler.enhance(image_bgr, outscale=float(upscale_factor))
    return output


# ---------------------------------------------------------------------------
# Core per-image processing helper
# ---------------------------------------------------------------------------

def process_single_image(
    image_bgr: np.ndarray,
    mode: str,
    upscale: float,
    sharpness: float,
    denoise: float,
    contrast: float,
    progress_fn=None
) -> np.ndarray:
    """Enhance a single BGR image and return enhanced BGR result."""
    is_ai_mode = "AI" in mode
    if is_ai_mode:
        try:
            result_bgr = enhance_ai_realesrgan(
                image_bgr=image_bgr, upscale_factor=upscale, progress_fn=progress_fn
            )
        except Exception as ai_err:
            logging.warning(f"AI Mode failed: {ai_err}. Falling back to Quick mode.")
            result_bgr = enhance_quick_opencv(
                image_bgr=image_bgr, upscale_factor=upscale,
                sharpness=sharpness, denoise_strength=denoise, contrast=contrast
            )
    else:
        result_bgr = enhance_quick_opencv(
            image_bgr=image_bgr, upscale_factor=upscale,
            sharpness=sharpness, denoise_strength=denoise, contrast=contrast
        )
    return result_bgr


# ---------------------------------------------------------------------------
# Gradio Batch Handler
# ---------------------------------------------------------------------------

def process_batch(
    uploaded_files,
    mode: str,
    upscale: float,
    sharpness: float,
    denoise: float,
    contrast: float,
    base_filename: str = "enhanced",
    progress=gr.Progress()
):
    """
    Batch processing entry point.
    Accepts multiple uploaded files, enhances each one, returns:
    - gallery of (enhanced_rgb_array, caption) tuples
    - zip file path for bulk download
    - status HTML string
    """
    if not uploaded_files or len(uploaded_files) == 0:
        raise gr.Error(
            "Kripya pehle ek ya adhik images upload karein! / Please upload at least one image!"
        )

    clean_base = re.sub(r'[\\/*?:"<>|\r\n\t]', "", str(base_filename).strip())
    if clean_base.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        clean_base = os.path.splitext(clean_base)[0]
    if not clean_base:
        clean_base = "enhanced"

    temp_dir = tempfile.mkdtemp()
    gallery_images = []
    saved_paths = []
    errors = []
    total = len(uploaded_files)

    for idx, file_obj in enumerate(uploaded_files):
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        original_name = os.path.splitext(os.path.basename(file_path))[0]

        progress(
            idx / total,
            desc=f"Processing image {idx + 1}/{total}: {os.path.basename(file_path)}"
        )

        try:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                errors.append(f"Could not read: {os.path.basename(file_path)}")
                continue

            result_bgr = process_single_image(
                img_bgr, mode, upscale, sharpness, denoise, contrast
            )

            out_filename = f"{clean_base}_{idx + 1:02d}_{original_name}.jpg"
            out_path = os.path.join(temp_dir, out_filename)
            cv2.imwrite(out_path, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            saved_paths.append(out_path)

            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            gallery_images.append((result_rgb, f"Image {idx + 1}: {original_name}"))

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}", exc_info=True)
            errors.append(f"{os.path.basename(file_path)}: {str(e)}")

    progress(1.0, desc="All images processed!")

    zip_path = None
    if saved_paths:
        zip_path = os.path.join(temp_dir, f"{clean_base}_enhanced_batch.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for sp in saved_paths:
                zf.write(sp, os.path.basename(sp))

    success_count = len(saved_paths)
    fail_count = len(errors)
    status_parts = [
        "<div class=\'status-box\'>",
        f"<span class=\'status-success\'>✅ {success_count} image(s) enhanced successfully</span>",
    ]
    if fail_count > 0:
        status_parts.append(
            f"<span class=\'status-error\'>&nbsp;&nbsp;|&nbsp;&nbsp;⚠️ {fail_count} failed</span>"
        )
        for err in errors:
            status_parts.append(f"<p class=\'err-detail\'>❌ {err}</p>")
    status_parts.append("</div>")
    status_html = "\n".join(status_parts)

    return gallery_images, zip_path, status_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; background: #0a0f1e !important; }
body, .gradio-container { background: #0a0f1e !important; }
.header-box {
    text-align: center; padding: 36px 20px 28px; margin-bottom: 20px; border-radius: 18px;
    background: linear-gradient(135deg, #0d1b4b 0%, #0a0f1e 50%, #12062e 100%);
    border: 1px solid rgba(99,102,241,0.25);
    box-shadow: 0 0 60px rgba(99,102,241,0.12), 0 4px 24px rgba(0,0,0,0.5);
    position: relative; overflow: hidden;
}
.header-box::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 30%, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.header-box h1 {
    margin: 0 0 10px; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.header-box p { margin: 0; color: rgba(203,213,225,0.75); font-size: 1rem; line-height: 1.5; }
.badge {
    display: inline-block; background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35); border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; color: #a5b4fc; margin-top: 10px; letter-spacing: 0.5px;
}
.enhance-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: white !important; font-size: 1.05rem !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 14px 28px !important; border: none !important;
    width: 100% !important; margin-top: 10px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s ease !important; letter-spacing: 0.3px !important;
}
.enhance-btn:hover { box-shadow: 0 6px 28px rgba(99,102,241,0.6) !important; transform: translateY(-1px) !important; }
.status-box { padding: 12px 18px; border-radius: 10px; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); margin-top: 8px; }
.status-success { color: #34d399; font-weight: 600; font-size: 0.95rem; }
.status-error   { color: #f87171; font-weight: 600; font-size: 0.95rem; }
.err-detail     { color: #fca5a5; font-size: 0.82rem; margin: 4px 0 0 12px; }
.info-tip {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px; padding: 10px 14px; color: #94a3b8; font-size: 0.82rem;
    line-height: 1.5; margin-bottom: 12px;
}
"""


def create_ui():
    """Build and configure the Gradio web interface for batch image enhancement."""
    with gr.Blocks(
        title="✨ Batch AI Image Enhancer",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0a0f1e",
            body_text_color="#cbd5e1",
            block_background_fill="rgba(15,23,42,0.7)",
            block_border_color="rgba(99,102,241,0.2)",
            block_label_text_color="#94a3b8",
            input_background_fill="rgba(15,23,42,0.9)",
            slider_color="#6366f1",
        )
    ) as demo:

        gr.HTML("""
        <div class="header-box">
            <h1>✨ Batch AI Image Enhancer</h1>
            <p>Ek saath kai images upload karein aur unhe ek click mein enhance karein.<br>
               Upload multiple images and enhance them all at once with AI &amp; OpenCV.</p>
            <span class="badge">🔒 100% Local &nbsp;·&nbsp; No Cloud &nbsp;·&nbsp; No API Keys</span>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left Column: Controls ──────────────────────────────────────
            with gr.Column(scale=1, min_width=340):

                gr.HTML(
                    "<div class='info-tip'>📁 <strong>Multiple Upload:</strong> "
                    "Ctrl/Cmd+Click se multiple files select karein ya drag &amp; drop karein. "
                    "Sab images ek saath enhance hongi aur ZIP mein milenge.</div>"
                )

                input_files = gr.File(
                    label="📤 Upload Images (Ctrl+Click for Multiple)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    elem_id="batch_file_uploader",
                    height=160,
                )

                mode_selector = gr.Radio(
                    choices=[
                        "AI (Real-ESRGAN) - best quality",
                        "Quick (OpenCV) - fast",
                    ],
                    value="Quick (OpenCV) - fast",
                    label="🧠 Enhancement Mode",
                    info="AI: Deep learning super-resolution | Quick: OpenCV filters (recommended for batch)"
                )

                upscale_slider = gr.Slider(
                    minimum=1.0, maximum=4.0, value=2.0, step=0.5,
                    label="🔍 Upscale Factor",
                    info="1x = original  ·  2x = double  ·  4x = 4× resolution"
                )

                with gr.Accordion("⚙️ Quick Mode Fine-Tuning Controls", open=False):
                    sharpness_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.8, step=0.1,
                        label="🔪 Sharpness (Unsharp Mask)",
                        info="Edge contrast & clarity boost"
                    )
                    denoise_slider = gr.Slider(
                        minimum=0.0, maximum=20.0, value=3.0, step=1.0,
                        label="🌫️ Denoise Strength",
                        info="Removes sensor noise and grain"
                    )
                    contrast_slider = gr.Slider(
                        minimum=0.0, maximum=4.0, value=1.5, step=0.1,
                        label="🌗 Contrast (CLAHE)",
                        info="Adaptive histogram equalization on luminance"
                    )

                filename_input = gr.Textbox(
                    label="🏷️ Output Base Filename",
                    value="enhanced",
                    placeholder="e.g. portrait, wallpaper, batch",
                    info="Files saved as: <name>_01_img.jpg, <name>_02_img.jpg ..."
                )

                enhance_btn = gr.Button(
                    "🚀 Enhance All Images",
                    variant="primary",
                    elem_classes=["enhance-btn"]
                )

                status_output = gr.HTML(value="", elem_id="status_display")

            # ── Right Column: Results ──────────────────────────────────────
            with gr.Column(scale=2, min_width=500):

                gallery_output = gr.Gallery(
                    label="🖼️ Enhanced Images Preview",
                    show_label=True,
                    elem_id="results_gallery",
                    columns=3,
                    rows=2,
                    height=500,
                    object_fit="contain",
                    preview=True,
                )

                zip_download = gr.File(
                    label="📦 Download All Enhanced Images (ZIP)",
                    interactive=False,
                    elem_id="zip_download_btn",
                    height=80,
                )

        enhance_btn.click(
            fn=process_batch,
            inputs=[
                input_files,
                mode_selector,
                upscale_slider,
                sharpness_slider,
                denoise_slider,
                contrast_slider,
                filename_input,
            ],
            outputs=[gallery_output, zip_download, status_output]
        )

    return demo


def get_free_port(preferred_port: int = 7860) -> int:
    """Check if preferred_port is available; if not, return an open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    demo_app = create_ui()
    port = get_free_port(7860)
    print("\n=======================================================")
    # Keep startup output compatible with the default Windows cp1252 console.
    print(f"  Batch Image Enhancer running at: http://127.0.0.1:{port}")
    print("=======================================================\n")
    demo_app.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        inbrowser=True
    )
