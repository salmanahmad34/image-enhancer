# 🌟 Local Image Enhancer & Upscaler

A 100% local, privacy-friendly AI image enhancement web app built with **Gradio**, **Real-ESRGAN**, and **OpenCV**. 

No external paid APIs, no subscriptions, and no data leaves your computer. Everything runs locally on your machine.

---

## 🎯 Features

- **AI Super-Resolution (Real-ESRGAN)**: Uses deep learning (`RealESRGAN_x4plus` with `RRDBNet`) to restore fine details, sharp textures, and upscale images up to 4x.
- **Quick Mode (OpenCV)**: High-speed, lightweight classical computer vision pipeline featuring:
  - Non-Local Means Denoising (`fastNlMeansDenoisingColored`)
  - Lanczos 4-lobe interpolation upscaling
  - Two-pass Unsharp Masking (micro-radius & macro-radius)
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space
- **Automatic Fallback Protection**: If AI mode cannot run (e.g. out of memory or missing GPU drivers), it gracefully warns you and automatically falls back to Quick mode.
- **Interactive UI**: Slider adjustments for upscale factor, sharpness, noise reduction, and contrast with immediate side-by-side preview and PNG download.

---

## 💻 System Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **RAM**: Minimum 4GB (8GB+ recommended)
- **GPU (Optional)**: NVIDIA GPU with CUDA support for accelerated AI inference. If no CUDA GPU is detected, the app automatically runs on CPU.

---

## 🚀 Installation & Setup

### 1. Windows (PowerShell or Command Prompt)

```powershell
# Navigate into the project folder
cd image_enhancer

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (Or in CMD: .\venv\Scripts\activate.bat)

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. macOS & Linux (Terminal)

```bash
# Navigate into the project folder
cd image_enhancer

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Web App

Make sure your virtual environment is activated, then run:

```bash
python app.py
```

Once started, open your web browser and navigate to:
👉 **`http://127.0.0.1:7860`**

---

## 📥 First-Run Model Download

- When you enhance an image using **AI (Real-ESRGAN)** for the first time, the application will automatically download the pre-trained weights file `RealESRGAN_x4plus.pth` (~67 MB) into the `weights/` directory.
- This download happens **only once**. Subsequent runs will use the cached local file offline.

---

## ⚡ Hardware & Performance (CPU vs. GPU)

| Hardware | Quick Mode (OpenCV) | AI Mode (Real-ESRGAN) |
| :--- | :--- | :--- |
| **NVIDIA GPU (CUDA)** | < 0.5 seconds | ~ 1–3 seconds |
| **Modern Multi-core CPU** | ~ 0.5–2 seconds | ~ 10–30 seconds |

*Note: For CPU inference, images with longest dimension > 1200px are automatically pre-scaled to ensure quick and smooth processing.*

---

## 🛠️ Common Errors & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`
- **Cause**: Newer versions of `torchvision` removed the `functional_tensor` module that `basicsr` references.
- **Fix**: `app.py` already includes an automatic monkey-patch shim at the top of the file to alias this module safely.

### 2. `AttributeError: _ARRAY_API not found` or `numpy.core.multiarray failed to import`
- **Cause**: NumPy 2.x breaking binary compatibility with compiled wheels built for NumPy 1.x.
- **Fix**: Ensure your environment uses NumPy 1.x:
  ```bash
  pip install "numpy<2.0.0"
  ```

### 3. Out of Memory (OOM) on GPU / CPU
- **Fix**: Lower the upscale slider (e.g., from 4x to 2x) or select **Quick (OpenCV)** mode for instant, lightweight processing.

### 4. PowerShell Script Execution Policy Error (`Activate.ps1 cannot be loaded`)
- **Fix**: Run PowerShell as Administrator or execute:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
