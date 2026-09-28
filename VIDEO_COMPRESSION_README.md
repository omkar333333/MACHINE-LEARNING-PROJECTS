# 🎥 AI-Powered Movie & Video Compressor

A highly optimized, hardware-accelerated Flask web application that compresses large movie and video files efficiently while preserving maximum visual quality. Designed specifically to leverage NVIDIA GPUs (NVENC) for blazing-fast compression speeds.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![NVIDIA NVENC](https://img.shields.io/badge/NVIDIA-NVENC_Accelerated-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/nvidia-video-codec-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- ⚡ **Hardware Accelerated**: Uses NVIDIA's `hevc_nvenc` (H.265 / HEVC) encoder to crunch massive video files in minutes rather than hours.
- 🎯 **Smart Bitrate Capping**: Analyzes source video metadata to mathematically guarantee significant file size reduction without accidental bitrate inflation.
- 🎨 **Modern Glassmorphic UI**: Responsive, sleek dark-mode web dashboard with real-time compression progress polling.
- 📊 **Compression History & Analytics**: SQLite database logging all past compressions, computation times, and total megabytes saved, complete with direct download buttons.
- 🛡️ **Safety Fallbacks**: Automatically degrades gracefully to CPU encoding (`libx265` / `libx264`) if hardware acceleration is unavailable.

---

## 🛠️ Prerequisites

- **Python 3.8+**
- **FFmpeg**: Must be installed and accessible in your system's `PATH`.
- **NVIDIA GPU** *(Recommended)*: RTX 20/30/40 series or GTX 16 series to leverage NVENC hardware encoding.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/omkar333333/AI-Powered_Video_Compression_Platform.git
cd AI-Powered_Video_Compression_Platform
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify FFmpeg & GPU Support
```bash
# Verify FFmpeg is accessible
ffmpeg -version

# Check if NVIDIA NVENC is available
ffmpeg -encoders | grep nvenc
```

### 5. Launch the Application
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:5000`

---

## 🏗️ Architecture & Pipeline

```text
[User Upload (MP4/MKV)]
         │
         ▼
[Metadata Extractor (ffprobe)] ──► Bitrate & Resolution Analysis
         │
         ▼
[Smart Parameter Calculator] ──► Optimal CRF / Target Bitrate
         │
         ▼
[NVENC Hardware Encoding] ──► H.265 Compression via GPU
         │
         ▼
[SQLite History Logger] ──► Space Saved & Download Link
```

---

## 👨‍💻 Author

**Omkar Mote**  
- 🎓 *B.E. in Artificial Intelligence & Data Science*  
- 🌐 [GitHub Profile](https://github.com/omkar333333)
