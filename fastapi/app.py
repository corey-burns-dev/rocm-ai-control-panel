from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
import requests
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import io, os
import numpy as np
import cv2
import hashlib
import math

app = FastAPI(title="Corey's ROCm AI Lab")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://localhost:5000", "http://localhost:3000", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global feature flags (simple in-memory toggles)
LLM_SVG_ENABLED = True

# --------- Chat via Ollama ----------
@app.post("/chat")
def chat(prompt: str = Form(...), model: str = Form("phi3:mini")):
    r = requests.post("http://ollama:11434/api/generate",
                      json={
                          "model": model,
                          "prompt": prompt,
                          "stream": False,
                          # Avoid lingering VRAM usage
                          "keep_alive": "0s",
                          # Keep responses snappy/lightweight
                          "options": {"num_predict": 256, "num_ctx": 2048}
                      })
    r.raise_for_status()
    return {"response": r.json().get("response", "")}

# --------- Stable Diffusion: txt2img ----------
@app.post("/generate-image")
def generate_image(
    prompt: str = Form(...), 
    negative_prompt: str = Form("blurry, bad quality, distorted, deformed, ugly, bad anatomy, low resolution, worst quality"),
    steps: int = Form(40), 
    cfg_scale: float = Form(7.5), 
    width: int = Form(768), 
    height: int = Form(768),
    sampler_name: str = Form("DPM++ 2M Karras"),
    batch_size: int = Form(1),
    # High-Res Fix options
    enable_hr: bool = Form(False),
    hr_scale: float = Form(1.5),
    hr_upscaler: str = Form("Latent (bicubic)"),
    denoising_strength: float = Form(0.35),
    hr_second_pass_steps: int = Form(0)
):
    payload = {
        "prompt": prompt, 
        "negative_prompt": negative_prompt,
        "steps": steps, 
        "cfg_scale": cfg_scale, 
        "width": width, 
        "height": height,
        "sampler_name": sampler_name,
        "batch_size": batch_size,
        "n_iter": 1,
        # High-Res Fix
        "enable_hr": enable_hr,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "denoising_strength": denoising_strength,
        "hr_second_pass_steps": hr_second_pass_steps,
        # Misc
        "tiling": False
    }
    r = requests.post("http://sd-webui:7860/sdapi/v1/txt2img", json=payload)
    r.raise_for_status()
    data = r.json()
    if "images" not in data or not data["images"]:
        return JSONResponse({"error": "No image generated"}, status_code=500)
    # Return first image as PNG stream
    img_b64 = data["images"][0]
    import base64
    img_bytes = base64.b64decode(img_b64.split(",", 1)[-1])
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

# --------- SD WebUI: models management ----------
@app.get("/models")
def list_models():
    """Return available SD checkpoints (titles) from SD WebUI."""
    r = requests.get("http://sd-webui:7860/sdapi/v1/sd-models")
    r.raise_for_status()
    models = r.json()
    # Return a compact view
    return [
        {
            "title": m.get("title"),
            "model_name": m.get("model_name"),
            "hash": m.get("sha256") or m.get("hash"),
        }
        for m in models
    ]

@app.post("/set-model")
def set_model(title: str = Form(...)):
    """Set the active SD checkpoint by its title (as shown in /models)."""
    r = requests.post("http://sd-webui:7860/sdapi/v1/options", json={"sd_model_checkpoint": title})
    r.raise_for_status()
    return {"ok": True, "model": title}

# --------- Image utility helpers ----------
def load_image(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGBA")

def pil_to_stream(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return StreamingResponse(buf, media_type=f"image/{fmt.lower()}")

# --------- Convert: format (PNG/JPEG/WebP) ----------
@app.post("/convert/format")
async def convert_format(fmt: str = Form(...), file: UploadFile = File(...)):
    img = load_image(file)
    fmt = fmt.upper()
    if fmt not in ["PNG", "JPEG", "WEBP"]:
        return JSONResponse({"error": "Unsupported format"}, status_code=400)
    # If JPEG, drop alpha
    if fmt == "JPEG":
        img = img.convert("RGB")
    return pil_to_stream(img, fmt)

# --------- Convert: resize ----------
@app.post("/convert/resize")
async def resize_image(width: int = Form(...), height: int = Form(...), keep_aspect: bool = Form(True), file: UploadFile = File(...)):
    img = load_image(file)
    if keep_aspect:
        img = ImageOps.contain(img, (width, height))
    else:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return pil_to_stream(img)

# --------- Convert: rotate ----------
@app.post("/convert/rotate")
async def rotate(deg: float = Form(...), expand: bool = Form(True), file: UploadFile = File(...)):
    img = load_image(file)
    img = img.rotate(deg, expand=expand, resample=Image.Resampling.BICUBIC)
    return pil_to_stream(img)

# --------- Convert: crop ----------
@app.post("/convert/crop")
async def crop(left: int = Form(...), top: int = Form(...), right: int = Form(...), bottom: int = Form(...), file: UploadFile = File(...)):
    img = load_image(file)
    img = img.crop((left, top, right, bottom))
    return pil_to_stream(img)

# --------- Convert: grayscale ----------
@app.post("/convert/grayscale")
async def grayscale(file: UploadFile = File(...)):
    img = load_image(file).convert("RGB")
    img = ImageOps.grayscale(img)
    return pil_to_stream(img)

# --------- Convert: blur/sharpen ----------
@app.post("/convert/blur")
async def blur(radius: float = Form(2.0), file: UploadFile = File(...)):
    img = load_image(file)
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return pil_to_stream(img)

@app.post("/convert/sharpen")
async def sharpen(amount: float = Form(1.0), file: UploadFile = File(...)):
    img = load_image(file)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150*amount), threshold=3))
    return pil_to_stream(img)

# --------- Convert: watermark ----------
@app.post("/convert/watermark")
async def watermark(text: str = Form(...), opacity: float = Form(0.3), file: UploadFile = File(...)):
    base = load_image(file).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # Basic font (system default); for custom font, mount a font file and use ImageFont.truetype
    font = ImageFont.load_default()
    w, h = base.size
    text_w, text_h = draw.textsize(text, font=font)
    pos = (w - text_w - 10, h - text_h - 10)
    draw.text(pos, text, font=font, fill=(255, 255, 255, int(255*opacity)))
    out = Image.alpha_composite(base, overlay)
    return pil_to_stream(out)

# --------- Convert: OpenCV operations (contrast, rotate by matrix) ----------
@app.post("/convert/opencv/contrast")
async def opencv_contrast(alpha: float = Form(1.3), beta: int = Form(10), file: UploadFile = File(...)):
    img = load_image(file).convert("RGB")
    arr = np.array(img)
    out = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)
    out_img = Image.fromarray(out)
    return pil_to_stream(out_img)

@app.post("/convert/opencv/rotate")
async def opencv_rotate(deg: float = Form(...), file: UploadFile = File(...)):
    img = load_image(file).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    out = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out_img = Image.fromarray(out)
    return pil_to_stream(out_img)

# --------- Pipeline: generate -> resize -> format ----------
@app.post("/pipeline/generate-resize-format")
def pipeline_generate_resize_format(prompt: str = Form(...), width: int = Form(512), height: int = Form(512), fmt: str = Form("PNG")):
    # Generate
    r = requests.post("http://sd-webui:7860/sdapi/v1/txt2img",
                      json={"prompt": prompt, "width": width, "height": height})
    r.raise_for_status()
    data = r.json()
    if "images" not in data or not data["images"]:
        return JSONResponse({"error": "No image generated"}, status_code=500)
    import base64
    img_bytes = base64.b64decode(data["images"][0].split(",", 1)[-1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    # Resize (contain) then format
    img = ImageOps.contain(img, (width, height))
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
    return pil_to_stream(img, fmt.upper())

# --------- GPU demo: matrix multiply (uses CPU if no CUDA/HIP) ----------
@app.post("/analyze/matrix-mul")
def matrix_mul(n: int = Form(1000)):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        c = torch.matmul(a, b)
        return {"device": device, "result_shape": list(c.shape)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --------- Sysadmin Assistant ----------
OLLAMA_URL = "http://ollama:11434/api/generate"

@app.post("/sysadmin")
def sysadmin(request: str = Form(...)):
    """
    Take a natural language sysadmin request and return a suggested shell command or config snippet.
    """
    prompt = f"""
    You are a Linux sysadmin assistant. 
    The user will describe a task in plain English. 
    Respond ONLY with the most relevant shell command(s) or config snippet, no extra explanation.

    Task: {request}
    """

    r = requests.post(OLLAMA_URL,
                      json={
                          "model": "phi3:mini",
                          "prompt": prompt,
                          "stream": False,
                          "keep_alive": "0s",
                          "options": {"num_predict": 200, "num_ctx": 2048}
                      })
    r.raise_for_status()
    return {"suggestion": r.json().get("response", "").strip()}

@app.post("/ollama/stop")
def ollama_stop_all():
    """Stop/unload all running Ollama models to free GPU/VRAM quickly."""
    try:
        # Query running models
        ps = requests.get("http://ollama:11434/api/ps", timeout=2)
        names = []
        if ps.ok:
            data = ps.json()
            for m in data.get("models", []):
                n = m.get("name") or m.get("model")
                if n:
                    names.append(n)
        # Stop each model
        for n in names:
            try:
                requests.post("http://ollama:11434/api/stop", json={"name": n}, timeout=2)
            except Exception:
                pass
        return {"stopped": names}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --------- SVG Logo Generator (LLM-driven) ----------
@app.post("/generate-svg")
def generate_svg(
    brand: str = Form(...),
    slogan: str = Form(""),
    style: str = Form("minimal"),
    primary_color: str = Form("#1e90ff"),
    secondary_color: str = Form("#111111"),
    prompt: str = Form(""),
    temperature: float = Form(0.3),
    size: int = Form(512),
    speed: str = Form("fast"),  # fast | balanced | detailed
    keep_warm_secs: int = Form(60)
):
    """
    Use the LLM to produce a single self-contained SVG logo.
    Returns: image/svg+xml
    """
    # Server kill switch to prevent Ollama from being used for SVG
    global LLM_SVG_ENABLED
    if not LLM_SVG_ENABLED:
        return JSONResponse({"error": "LLM SVG generation is disabled by server settings"}, status_code=403)
    # Speed presets: trade detail for latency
    if speed not in ("fast", "balanced", "detailed"):
        speed = "fast"

    if speed == "fast":
        num_predict = 360
        extra_rules = "Avoid gradients, filters, masks. Use <=5 shapes/paths. Keep under 8KB."
    elif speed == "detailed":
        num_predict = 900
        extra_rules = "Gradients allowed sparingly. Prefer vectors, still concise (<25KB)."
    else:  # balanced
        num_predict = 600
        extra_rules = "No heavy filters. Keep concise (<16KB)."

    sys_instructions = f"""
You are a professional logo designer who outputs only valid inline SVG.
Generate a {style} logo for the brand "{brand}"{(' with slogan "' + slogan + '"') if slogan else ''}.
Requirements:
- Use only vector elements (paths/shapes), no raster images.
- Palette: primary {primary_color}, secondary {secondary_color}. You may compute tints/shades.
- Include viewBox="0 0 {size} {size}" and width/height="{size}".
- Center composition, balanced whitespace.
- If using text, prefer generic fonts (e.g., 'Montserrat, Arial, sans-serif').
- {extra_rules}
- Return ONLY the SVG markup, beginning with <svg and ending with </svg>.
Extra creative guidance: {prompt}
"""

    # Keep the model warm briefly to avoid reloading on successive generations
    ka = f"{keep_warm_secs}s" if keep_warm_secs and keep_warm_secs > 0 else "0s"
    req = {
        "model": "phi3:mini",
        "prompt": sys_instructions,
        "stream": False,
        "keep_alive": ka,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 2048
        }
    }
    r = requests.post(OLLAMA_URL, json=req)
    r.raise_for_status()
    text = r.json().get("response", "")

    # Strip code fences and extract <svg>...</svg>
    if "```" in text:
        # Remove markdown code fences
        text = text.replace("```svg", "").replace("```xml", "").replace("```", "").strip()

    lower = text.lower()
    start = lower.find("<svg")
    end = lower.rfind("</svg>")
    if start != -1 and end != -1:
        svg = text[start:end+6]
    else:
        svg = text

    # Basic sanity guard
    if "<svg" not in svg:
        return JSONResponse({"error": "LLM did not return SVG"}, status_code=502)

    return Response(content=svg, media_type="image/svg+xml")


# --------- SVG Logo Generator (Algorithmic, instant) ----------
@app.post("/generate-svg-fast")
def generate_svg_fast(
        brand: str = Form(...),
        slogan: str = Form(""),
        style: str = Form("minimal"),
        primary_color: str = Form("#1e90ff"),
        secondary_color: str = Form("#111111"),
        size: int = Form(512)
):
        """
        Generate a clean, deterministic logo SVG without using the LLM.
        Uses geometric shapes based on a hash of the inputs for variety.
        """
        seed_src = f"{brand}|{slogan}|{style}|{primary_color}|{secondary_color}"
        h = hashlib.sha256(seed_src.encode()).digest()
        rnd = h[0]
        # Choose polygon sides: 3..8
        sides = 3 + (rnd % 6)
        rotation = (h[1] / 255.0) * math.pi
        stroke_w = max(2, size // 24)
        radius = int(size * (0.28 + (h[2] / 255.0) * 0.08))

        # Build polygon path
        pts = []
        for i in range(sides):
                ang = rotation + (2 * math.pi * i / sides)
                x = math.cos(ang) * radius
                y = math.sin(ang) * radius
                pts.append((x, y))
        d = "M " + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"

        # Optional inner mark
        inner_ratio = 0.5 + (h[3] / 255.0) * 0.2
        inner_r = radius * inner_ratio
        pts2 = []
        for i in range(sides):
                ang = rotation + (2 * math.pi * (i + 0.5) / sides)
                x = math.cos(ang) * inner_r
                y = math.sin(ang) * inner_r
                pts2.append((x, y))
        d2 = "M " + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts2) + " Z"

        # Colors and text
        bg_rx = size // 10
        text_color = "#ffffff"
        brand_y = size - size // 8
        slogan_y = brand_y + size // 18

        svg = f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{primary_color}"/>
            <stop offset="100%" stop-color="{secondary_color}"/>
        </linearGradient>
    </defs>
    <rect width="{size}" height="{size}" rx="{bg_rx}" fill="url(#g)"/>
    <g fill="none" stroke="{text_color}" stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round" transform="translate({size//2},{size//2})">
        <path d="{d}"/>
        <path d="{d2}" opacity="0.7"/>
    </g>
    <g fill="{text_color}" text-anchor="middle">
        <text x="{size//2}" y="{brand_y}" font-size="{size//14}" font-family="Montserrat, Arial, sans-serif" font-weight="600">{brand}</text>
        {f'<text x="{size//2}" y="{slogan_y}" font-size="{size//24}" font-family="Montserrat, Arial, sans-serif" opacity="0.8">{slogan}</text>' if slogan else ''}
    </g>
</svg>
"""
        return Response(content=svg.strip(), media_type="image/svg+xml")


# --------- Settings: Enable/Disable LLM SVG ----------
@app.get("/settings/llm-svg-enabled")
def get_llm_svg_enabled():
    return {"enabled": LLM_SVG_ENABLED}

@app.post("/settings/llm-svg-enabled")
def set_llm_svg_enabled(enabled: bool = Form(...)):
    global LLM_SVG_ENABLED
    LLM_SVG_ENABLED = bool(enabled)
    return {"enabled": LLM_SVG_ENABLED}



