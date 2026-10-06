"""
Test script to generate a synthetic blurry/noisy image and verify
both Quick (OpenCV) and AI (Real-ESRGAN) enhancement pipelines, plus fallback.
"""

import os
import sys

# Ensure local venv site-packages and app directory are always discoverable
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_SITE_PACKAGES = os.path.join(_CURRENT_DIR, "venv", "Lib", "site-packages")
if os.path.isdir(_VENV_SITE_PACKAGES) and _VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, _VENV_SITE_PACKAGES)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import numpy as np  # type: ignore
import cv2  # type: ignore

# Import from app.py
from app import enhance_quick_opencv, enhance_ai_realesrgan, process_image, REALESRGAN_AVAILABLE


def create_synthetic_blurry_image(path: str = "test_blurry.png") -> np.ndarray:
    """Generate a synthetic test image with text, shapes, blur, and noise."""
    # Create a 200x200 RGB image with gradient & patterns
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            img[i, j] = [(i * 2) % 256, (j * 2) % 256, ((i + j) * 3) % 256]

    # Draw sharp geometric shapes and text
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    cv2.rectangle(img, (30, 30), (80, 80), (0, 255, 0), -1)
    cv2.putText(img, "TEST 123", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Downscale and apply GaussianBlur to simulate low-res blurry photo
    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    blurry = cv2.GaussianBlur(small, (7, 7), sigmaX=2.0)

    # Add Gaussian noise
    noise = np.random.normal(0, 8, blurry.shape).astype(np.int16)
    noisy_blurry = np.clip(blurry.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, noisy_blurry)
    print(f"Created synthetic blurry test image at: {path} (shape: {noisy_blurry.shape})")
    return noisy_blurry


def test_pipelines():
    test_img_bgr = create_synthetic_blurry_image("test_blurry_input.png")
    test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)

    print("\n--- Testing Quick Mode (OpenCV) ---")
    quick_out = enhance_quick_opencv(
        image_bgr=test_img_bgr,
        upscale_factor=2.0,
        sharpness=0.8,
        denoise_strength=3.0,
        contrast=1.5
    )
    assert quick_out.shape[0] == 200 and quick_out.shape[1] == 200, f"Unexpected shape: {quick_out.shape}"
    cv2.imwrite("test_quick_output.png", quick_out)
    print(f"Quick Mode Success! Output shape: {quick_out.shape}, saved to test_quick_output.png")

    print("\n--- Testing Full process_image Gradio Handler (Quick Mode + Custom Name) ---")
    result_rgb, download_path = process_image(
        input_img=test_img_rgb,
        mode="Quick (OpenCV) - fast",
        upscale=2.0,
        sharpness=0.8,
        denoise=3.0,
        contrast=1.5,
        custom_filename="creative"
    )
    assert os.path.exists(download_path), "Download file was not created"
    assert download_path.endswith(".jpg"), f"Expected .jpg extension, got {download_path}"
    assert "creative.jpg" in download_path, f"Expected custom name 'creative.jpg' in {download_path}"
    print(f"process_image (Quick) Success! Preview shape: {result_rgb.shape}, file: {download_path}")

    print("\n--- Testing AI Mode (Real-ESRGAN) or Fallback with Custom Filename ---")
    try:
        ai_out_rgb, ai_file = process_image(
            input_img=test_img_rgb,
            mode="AI (Real-ESRGAN) - best quality",
            upscale=2.0,
            sharpness=0.8,
            denoise=3.0,
            contrast=1.5,
            custom_filename="my_enhanced_artwork"
        )
        assert os.path.exists(ai_file), "AI download file not created"
        assert ai_file.endswith(".jpg"), f"Expected .jpg extension, got {ai_file}"
        assert "my_enhanced_artwork.jpg" in ai_file, f"Expected 'my_enhanced_artwork.jpg' in {ai_file}"
        print(f"AI/Fallback Mode processed successfully! Shape: {ai_out_rgb.shape}, file: {ai_file}")
    except Exception as e:
        print(f"AI Mode test error: {e}")

    print("\n=== All programmatic tests completed! ===")


if __name__ == "__main__":
    test_pipelines()
