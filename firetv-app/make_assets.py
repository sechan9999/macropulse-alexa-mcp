"""Generates the Fire TV app's icon/banner (bundled in the APK) and the Amazon Appstore listing images,
in the same style as the Alexa+ add-on assets (addon-package/make_assets.py). Needs Pillow.

    python firetv-app/make_assets.py

  assets/icon.png            1024x1024  app icon (app.json "icon")
  assets/tv-banner.png        320x180   Android TV / Fire TV launcher banner (config-tv androidTVBanner)
  store/icon-512.png          512x512   Appstore listing: large icon
  store/icon-114.png          114x114   Appstore listing: small icon
  store/firetv-icon-1280x720.png        Appstore listing: Fire TV app icon (tile)
  store/firetv-background-1920x1080.png Appstore listing: Fire TV background image
  store/featured-logo-640x260.png       Appstore listing: featured content logo (transparent)
  store/featured-background-1920x720.png Appstore listing: featured content background
Screenshots (1920x1080) must be real captures from a device/emulator, e.g.:
  adb shell screencap -p /sdcard/s1.png && adb pull /sdcard/s1.png
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "addon-package"))
from make_assets import FONT, FONT_BOLD, MUTED, TEAL, WHITE, font, gradient, icon, trend  # noqa: E402
from PIL import Image, ImageChops, ImageDraw, ImageFilter  # noqa: E402


def wide(w: int, h: int, tagline: bool) -> Image.Image:
    """Landscape tile: trend motif left, wordmark right (banner, Fire TV icon, background)."""
    s = 2 if w < 1000 else 1                       # supersample the small banner for clean edges
    W, H = w * s, h * s
    img = gradient(W, H)
    d = ImageDraw.Draw(img)
    trend(d, (int(W * 0.07), int(H * 0.28), int(W * 0.38), int(H * 0.72)),
          width=max(int(H * 0.035), 2), dot=max(int(H * 0.03), 2))
    x = int(W * 0.44)
    room = W * 0.94 - x

    def fit(paths, size, text):                    # largest size <= `size` whose text fits the room
        while size > 8 and d.textlength(text, font=font(paths, size)) > room:
            size -= 1
        return font(paths, size)

    title = fit(FONT_BOLD, int(H * (0.19 if tagline else 0.24)), "MacroPulse")
    d.text((x, int(H * (0.30 if tagline else 0.36))), "MacroPulse", font=title, fill=WHITE)
    if tagline:
        line1, line2 = "Live macro signals on your TV", "Informational only, not investment advice."
        d.text((x, int(H * 0.53)), line1, font=fit(FONT, int(H * 0.065), line1), fill=TEAL)
        d.text((x, int(H * 0.64)), line2, font=fit(FONT, int(H * 0.04), line2), fill=MUTED)
    return img.resize((w, h), Image.LANCZOS) if s > 1 else img


# Price path for the featured artwork: fractions of the chart box (x, y-up), gently rising with pullbacks.
PATH = [(0.00, 0.18), (0.08, 0.26), (0.15, 0.21), (0.24, 0.36), (0.32, 0.31), (0.41, 0.47), (0.49, 0.42),
        (0.58, 0.58), (0.66, 0.52), (0.75, 0.70), (0.83, 0.66), (0.92, 0.84), (1.00, 0.92)]


def rounded_mask(size, radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return m


def featured_logo(w: int = 640, h: int = 260) -> Image.Image:
    """Self-contained navy badge (readable on the white console preview and on any Fire TV row):
    icon tile, wordmark and tagline. Corners are transparent (transparency is optional here)."""
    s = 3
    W, H = w * s, h * s
    badge = gradient(W, H).convert("RGBA")
    d = ImageDraw.Draw(badge)
    tile = int(H * 0.56)                                   # rounded icon tile on the left
    tx, ty = int(H * 0.22), (H - tile) // 2
    d.rounded_rectangle((tx, ty, tx + tile, ty + tile), radius=int(tile * 0.22), fill=(28, 52, 88))
    pad = tile * 0.2
    trend(d, (tx + pad, ty + pad, tx + tile - pad, ty + tile - pad), width=int(tile * 0.075), dot=int(tile * 0.06))
    x = tx + tile + int(H * 0.14)
    room = W - x - int(H * 0.18)
    size = int(H * 0.30)
    while d.textlength("MacroPulse", font=font(FONT_BOLD, size)) > room:
        size -= 1
    title = font(FONT_BOLD, size)
    d.text((x, int(H * 0.26)), "MacroPulse", font=title, fill=WHITE)
    tag = "MACRO SIGNALS ON YOUR TV"
    tsize = int(size * 0.34)
    while d.textlength(tag, font=font(FONT_BOLD, tsize)) > room:
        tsize -= 1
    d.text((x + int(size * 0.04), int(H * 0.26) + int(size * 1.22)), tag, font=font(FONT_BOLD, tsize), fill=TEAL)
    badge.putalpha(rounded_mask((W, H), int(H * 0.16)))
    return badge.resize((w, h), Image.LANCZOS)


def featured_background(w: int = 1920, h: int = 720) -> Image.Image:
    """Text-free backdrop (Fire TV draws the logo over its left side): faint grid, a glowing teal
    area chart on the right, and a darker left edge so the logo stays readable."""
    s = 2
    W, H = w * s, h * s
    img = gradient(W, H).convert("RGBA")

    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))     # faint chart grid
    g = ImageDraw.Draw(grid)
    step = 60 * s
    for gx in range(0, W, step):
        g.line((gx, 0, gx, H), fill=(120, 150, 190, 18), width=s)
    for gy in range(0, H, step):
        g.line((0, gy, W, gy), fill=(120, 150, 190, 18), width=s)
    img.alpha_composite(grid)

    x0, x1, y0, y1 = W * 0.42, W * 0.95, H * 0.18, H * 0.86
    pts = [(x0 + (x1 - x0) * fx, y1 - (y1 - y0) * fy) for fx, fy in PATH]

    area = Image.new("RGBA", (W, H), (0, 0, 0, 0))     # teal fill fading toward the bottom
    ImageDraw.Draw(area).polygon(pts + [(x1, H), (x0, H)], fill=TEAL + (255,))
    fade = Image.new("L", (1, H))
    for y in range(H):
        fade.putpixel((0, y), int(70 * max(0.0, 1 - (y - y0) / (H - y0)) ** 1.4) if y >= y0 else 70)
    alpha = ImageChops.multiply(area.getchannel("A"), fade.resize((W, H)))
    edge = Image.new("L", (W, 1))                       # and fading in from the chart's left edge
    for x in range(W):
        edge.putpixel((x, 0), int(255 * min(1.0, max(0.0, (x - x0) / ((x1 - x0) * 0.25)))))
    alpha = ImageChops.multiply(alpha, edge.resize((W, H)))
    area.putalpha(alpha)
    img.alpha_composite(area)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))     # soft glow under the line
    ImageDraw.Draw(glow).line(pts, fill=TEAL + (160,), width=int(H * 0.035), joint="curve")
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(H * 0.03)))

    d = ImageDraw.Draw(img)
    d.line(pts, fill=TEAL, width=int(H * 0.012), joint="curve")
    cx, cy = pts[-1]
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    r = H * 0.05
    ImageDraw.Draw(halo).ellipse((cx - r, cy - r, cx + r, cy + r), fill=TEAL + (120,))
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(H * 0.02)))
    r = H * 0.016
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)

    shade = Image.new("L", (W, 1))                      # darken the left 45% for the logo overlay
    for x in range(W):
        shade.putpixel((x, 0), int(150 * max(0.0, 1 - x / (W * 0.45)) ** 1.5))
    left = Image.new("RGBA", (W, H), (8, 16, 30, 255))
    left.putalpha(shade.resize((W, H)))
    img.alpha_composite(left)
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


def main() -> None:
    (HERE / "assets").mkdir(exist_ok=True)
    (HERE / "store").mkdir(exist_ok=True)
    icon(1024).save(HERE / "assets/icon.png", optimize=True)
    wide(320, 180, tagline=False).save(HERE / "assets/tv-banner.png", optimize=True)
    icon(512).save(HERE / "store/icon-512.png", optimize=True)
    icon(114).save(HERE / "store/icon-114.png", optimize=True)
    wide(1280, 720, tagline=False).save(HERE / "store/firetv-icon-1280x720.png", optimize=True)
    wide(1920, 1080, tagline=True).save(HERE / "store/firetv-background-1920x1080.png", optimize=True)
    featured_logo().save(HERE / "store/featured-logo-640x260.png", optimize=True)
    featured_background().save(HERE / "store/featured-background-1920x720.png", optimize=True)
    for p in sorted([*HERE.glob("assets/*.png"), *HERE.glob("store/*.png")]):
        print(p.relative_to(HERE), Image.open(p).size)


if __name__ == "__main__":
    main()
