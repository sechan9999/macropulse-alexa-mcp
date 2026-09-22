"""Regenerates the Alexa+ add-on store images in ./assets (needs Pillow).

    python addon-package/make_assets.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "assets"
ICON_SIZES = (72, 64, 88, 126, 180, 241)
NAVY, NAVY2, TEAL, WHITE, MUTED = (12, 24, 43), (22, 42, 72), (45, 212, 191), (240, 245, 252), (152, 168, 192)
FONT_BOLD, FONT = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def gradient(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), NAVY)
    px = img.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        row = tuple(int(NAVY[i] + (NAVY2[i] - NAVY[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = row
    return img


def trend(draw: ImageDraw.ImageDraw, box, width: int, dot: int) -> None:
    """Rising step-line with a marker on the last point (the 'regime' motif)."""
    x0, y0, x1, y1 = box
    fx = lambda f: x0 + (x1 - x0) * f
    fy = lambda f: y1 - (y1 - y0) * f
    pts = [(fx(0.00), fy(0.10)), (fx(0.22), fy(0.42)), (fx(0.42), fy(0.30)), (fx(0.68), fy(0.68)), (fx(1.00), fy(0.92))]
    draw.line(pts, fill=TEAL, width=width, joint="curve")
    cx, cy = pts[-1]
    draw.ellipse((cx - dot, cy - dot, cx + dot, cy + dot), fill=WHITE)


def icon(size: int) -> Image.Image:
    s = 4  # supersample, then downscale for clean edges
    big = size * s
    img = gradient(big, big)
    d = ImageDraw.Draw(img)
    pad = big * 0.2
    trend(d, (pad, pad, big - pad, big - pad), width=max(int(big * 0.07), 2), dot=int(big * 0.055))
    return img.resize((size, size), Image.LANCZOS)


def carousel() -> Image.Image:
    w, h = 600, 900
    img = gradient(w, h)
    d = ImageDraw.Draw(img)
    d.text((44, 70), "MacroPulse", font=font(FONT_BOLD, 60), fill=WHITE)
    d.text((44, 148), "Live macro signals,\nspoken.", font=font(FONT, 34), fill=TEAL, spacing=6)
    trend(d, (44, 270, w - 44, 430), width=8, dot=11)
    d.text((44, 470), "TRY SAYING", font=font(FONT_BOLD, 20), fill=MUTED)
    phrases = ["What's the macro regime?", "Simulate a hawkish Fed shock.", "How risky is SPY right now?"]
    y = 512
    for p in phrases:
        d.rounded_rectangle((44, y, w - 44, y + 74), radius=16, fill=(30, 54, 90))
        d.text((66, y + 18), f"\u201c{p}\u201d", font=font(FONT, 27), fill=WHITE)
        y += 92
    d.text((44, 828), "Real market data. Informational only,\nnot investment advice.", font=font(FONT, 20), fill=MUTED, spacing=4)
    return img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for size in ICON_SIZES:
        icon(size).save(OUT / f"icon-{size}.png", optimize=True)
    carousel().save(OUT / "carousel-1.png", optimize=True)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
