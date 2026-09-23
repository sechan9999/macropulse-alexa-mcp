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
from PIL import Image, ImageDraw  # noqa: E402


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


def featured_logo(w: int = 640, h: int = 260) -> Image.Image:
    """Trend motif + wordmark on a transparent canvas; Fire TV overlays it on the featured background."""
    s = 2
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    H = h * s
    trend(d, (int(H * 0.08), int(H * 0.28), int(H * 0.92), int(H * 0.72)), width=int(H * 0.05), dot=int(H * 0.045))
    x, room = int(H * 1.05), w * s * 0.97 - H * 1.05
    size = int(H * 0.36)
    while d.textlength("MacroPulse", font=font(FONT_BOLD, size)) > room:
        size -= 1
    f = font(FONT_BOLD, size)
    top, bottom = d.textbbox((0, 0), "MacroPulse", font=f)[1::2]
    d.text((x, (H - (bottom - top)) // 2 - top), "MacroPulse", font=f, fill=WHITE)
    return img.resize((w, h), Image.LANCZOS)


def featured_background(w: int = 1920, h: int = 720) -> Image.Image:
    """Wide, text-free backdrop (the logo is drawn over it): gradient with the trend motif on the right."""
    img = gradient(w, h)
    trend(ImageDraw.Draw(img), (int(w * 0.55), int(h * 0.22), int(w * 0.93), int(h * 0.78)),
          width=int(h * 0.03), dot=int(h * 0.025))
    return img


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
