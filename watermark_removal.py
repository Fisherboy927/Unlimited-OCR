"""Remove light gray watermarks from image files.

The script accepts a single image path or a folder path. When a folder is
provided, it displays a numbered list of all images and asks which ones to
clean. After processing, it returns to the menu so you can keep selecting.
Cleaned images are written under the project's input directory. Single-file
inputs keep the same file name. Folder inputs are written to
input/<folder_name>/ and keep their relative folder structure.

Detection modes:

- ``band`` (default) — keep pixels whose gray level sits in
  ``[--band-lo, --band-hi]`` with low chroma.  This matches tiled mid-gray
  watermarks (e.g. diagonal ``d26889``) without swallowing near-white paper.
  Dark text/lines and saturated stamps/ink are protected.
- ``bright`` — legacy gate: ``channel_min >= --threshold`` and low chroma,
  then connected-component area filtering.  Works for sparse bright speckles,
  not for dense tiled watermarks on white paper.

Fill modes:

- Default: paint masked pixels to paper white (``--fill-value``).
- ``--inpaint``: OpenCV Telea inpainting (can leave edge ghosts on dense
  watermarks; prefer band + white fill for those).

Optional ``--blackhat`` tightens the band mask to watermark strokes via
morphological black-hat.  ``--preview`` / ``--preview-only`` write a red
mask overlay next to the output.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "input"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove light gray watermarks from images.")
    parser.add_argument(
        "--path",
        "-p",
        help="Path to an image file or a folder. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--mode",
        choices=("band", "bright"),
        default="band",
        help="Watermark detection mode: 'band' (default) for mid-gray tiled "
        "watermarks, 'bright' for the legacy bright-pixel gate.",
    )
    parser.add_argument(
        "--band-lo",
        type=int,
        default=175,
        help="Lower gray bound for band mode (default: 175).",
    )
    parser.add_argument(
        "--band-hi",
        type=int,
        default=245,
        help="Upper gray bound for band mode (default: 245).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=210,
        help="Bright mode: pixels with channel_min >= this are candidates (default: 210).",
    )
    parser.add_argument(
        "--saturation",
        type=int,
        default=40,
        help="Maximum HSV saturation (0-255 scale) for a pixel to count as "
        "gray watermark (default: 40). Also used as channel-spread cap in bright mode.",
    )
    parser.add_argument(
        "--protect-dark",
        type=int,
        default=140,
        help="Band mode: pixels darker than this are treated as text/lines and "
        "excluded from the mask; a small dilation protects anti-aliased edges "
        "(default: 140).",
    )
    parser.add_argument(
        "--protect-sat",
        type=int,
        default=50,
        help="Band mode: pixels with HSV saturation above this are treated as "
        "stamps/ink and excluded (default: 50).",
    )
    parser.add_argument(
        "--dark-dilate",
        type=int,
        default=5,
        help="Band mode: kernel size for dilating dark-text protection "
        "(odd integer, default: 5).",
    )
    parser.add_argument(
        "--blackhat",
        action="store_true",
        help="Band mode: require morphological black-hat response so only "
        "stroke-like mid-gray pixels are kept.",
    )
    parser.add_argument(
        "--blackhat-ksize",
        type=int,
        default=31,
        help="Black-hat structuring-element size (odd integer, default: 31).",
    )
    parser.add_argument(
        "--blackhat-thresh",
        type=int,
        default=12,
        help="Minimum black-hat response to keep a pixel (default: 12).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=2,
        help="Ignore mask components smaller than this many pixels (default: 2). "
        "Set to 0 with a huge --max-area to disable filtering.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=12000,
        help="Ignore mask components larger than this many pixels (default: 12000). "
        "In band mode this mainly drops noise; paper is already excluded by the band.",
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        default=252,
        help="RGB value used when painting masked pixels white (default: 252).",
    )
    parser.add_argument(
        "--inpaint",
        action="store_true",
        help="Use OpenCV Telea inpainting instead of painting to --fill-value.",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=3,
        help="Radius for the inpainting operation (default: 3).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write the detected mask as a red overlay next to the output file.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only write the mask preview image; do not clean.",
    )
    return parser.parse_args()


def prompt_for_path() -> Path:
    while True:
        value = input("Image file or folder path: ").strip().strip('"').strip("'")
        if not value:
            print("Please enter a path.")
            continue

        path = Path(value).expanduser().resolve()
        if path.exists() and (path.is_file() or path.is_dir()):
            return path

        print(f"File or folder not found: {path}")


def is_supported_image(path: Path) -> bool:
    if path.name.startswith("._"):
        return False

    return path.suffix.lower() in IMAGE_SUFFIXES


def collect_image_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not is_supported_image(input_path):
            supported = ", ".join(sorted(IMAGE_SUFFIXES))
            raise ValueError(f"Unsupported image type '{input_path.suffix}'. Supported types: {supported}")
        return [input_path]

    image_files = sorted(path for path in input_path.rglob("*") if path.is_file() and is_supported_image(path))
    if not image_files:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"No supported image files found in {input_path}. Supported types: {supported}")

    return image_files


def output_root_for_input(input_path: Path) -> Path:
    if input_path.is_file():
        return OUTPUT_ROOT
    return OUTPUT_ROOT / input_path.name


def output_path_for_image(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        relative_path = Path(image_path.name)
    else:
        relative_path = Path(input_path.name) / image_path.relative_to(input_path)

    return OUTPUT_ROOT / relative_path


def load_image_libraries():
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "This script requires Pillow and NumPy. Install them with: pip install pillow numpy"
        ) from exc

    return Image, ImageOps, np


def load_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "This mode requires OpenCV. Install it with: pip install opencv-python-headless"
        ) from exc
    return cv2


def _filter_components(raw_mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Keep connected components whose area is in [min_area, max_area]."""
    import cv2

    if min_area <= 0 and max_area >= raw_mask.size:
        return raw_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    if num_labels <= 1:
        return raw_mask

    kept = np.zeros_like(raw_mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            kept[labels == label_id] = 255
    return kept


def detect_watermark_mask_bright(
    rgb: np.ndarray,
    threshold: int,
    saturation: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    """Legacy bright-pixel gate: channel_min >= threshold and low channel spread."""
    channel_min = rgb.min(axis=2)
    channel_max = rgb.max(axis=2)
    spread = channel_max.astype(np.int16) - channel_min.astype(np.int16)

    raw_mask = ((channel_min >= threshold) & (spread <= saturation)).astype(np.uint8) * 255
    return _filter_components(raw_mask, min_area, max_area)


def detect_watermark_mask_band(
    rgb: np.ndarray,
    *,
    band_lo: int,
    band_hi: int,
    saturation: int,
    protect_dark: int,
    protect_sat: int,
    dark_dilate: int,
    blackhat: bool,
    blackhat_ksize: int,
    blackhat_thresh: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    """Mid-gray band mask with text/stamp protection and optional black-hat.

    Pipeline:

    1. Intensity band — gray in ``[band_lo, band_hi]`` (watermark, not paper,
       not black ink).
    2. Low chroma — HSV saturation <= ``saturation``.
    3. Protect dark text/lines — dilate ``gray < protect_dark`` and exclude.
    4. Protect stamps/ink — exclude ``saturation > protect_sat``.
    5. Optional black-hat — keep only pixels with stroke-like response.
    6. Component area filter.
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    band = (gray >= band_lo) & (gray <= band_hi) & (sat <= saturation)

    dark = gray < protect_dark
    k = max(1, dark_dilate)
    if k % 2 == 0:
        k += 1
    near_dark = cv2.dilate(dark.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)

    colorful = sat > protect_sat
    candidate = band & ~near_dark & ~colorful

    if blackhat:
        bh_k = max(3, blackhat_ksize)
        if bh_k % 2 == 0:
            bh_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
        response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        candidate &= response >= blackhat_thresh

    raw_mask = candidate.astype(np.uint8) * 255
    # Light open to drop single-pixel JPEG speckles before component filter.
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _filter_components(raw_mask, min_area, max_area)


def detect_watermark_mask(
    rgb: np.ndarray,
    *,
    mode: str,
    threshold: int,
    saturation: int,
    min_area: int,
    max_area: int,
    band_lo: int,
    band_hi: int,
    protect_dark: int,
    protect_sat: int,
    dark_dilate: int,
    blackhat: bool,
    blackhat_ksize: int,
    blackhat_thresh: int,
) -> np.ndarray:
    """Return a uint8 mask (0/255) of pixels that look like watermark."""
    if mode == "bright":
        return detect_watermark_mask_bright(rgb, threshold, saturation, min_area, max_area)

    return detect_watermark_mask_band(
        rgb,
        band_lo=band_lo,
        band_hi=band_hi,
        saturation=saturation,
        protect_dark=protect_dark,
        protect_sat=protect_sat,
        dark_dilate=dark_dilate,
        blackhat=blackhat,
        blackhat_ksize=blackhat_ksize,
        blackhat_thresh=blackhat_thresh,
        min_area=min_area,
        max_area=max_area,
    )


def remove_light_gray_watermark(
    image_path: Path,
    output_path: Path,
    *,
    mode: str,
    threshold: int,
    saturation: int,
    min_area: int,
    max_area: int,
    band_lo: int,
    band_hi: int,
    protect_dark: int,
    protect_sat: int,
    dark_dilate: int,
    blackhat: bool,
    blackhat_ksize: int,
    blackhat_thresh: int,
    fill_value: int,
    inpaint: bool,
    inpaint_radius: int,
    preview: bool,
    preview_only: bool,
) -> None:
    Image, ImageOps, np = load_image_libraries()
    cv2 = load_opencv()

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        array = np.array(image)
        has_alpha = array.ndim == 3 and array.shape[2] == 4
        rgb = array[..., :3].astype(np.uint8)

        mask = detect_watermark_mask(
            rgb,
            mode=mode,
            threshold=threshold,
            saturation=saturation,
            min_area=min_area,
            max_area=max_area,
            band_lo=band_lo,
            band_hi=band_hi,
            protect_dark=protect_dark,
            protect_sat=protect_sat,
            dark_dilate=dark_dilate,
            blackhat=blackhat,
            blackhat_ksize=blackhat_ksize,
            blackhat_thresh=blackhat_thresh,
        )

        if preview or preview_only:
            preview_image = rgb.copy()
            preview_image[mask == 255] = [255, 0, 0]
            preview_path = output_path.with_suffix(".mask-preview.png")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(preview_image).save(preview_path)
            print(f"    Mask preview: {preview_path}")

        if preview_only:
            return

        if inpaint:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cleaned_bgr = cv2.inpaint(bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        else:
            cleaned_rgb = rgb.copy()
            fill = int(np.clip(fill_value, 0, 255))
            cleaned_rgb[mask == 255] = [fill, fill, fill]

        if has_alpha:
            cleaned = np.dstack([cleaned_rgb, array[..., 3]])
            output_image = Image.fromarray(cleaned, mode="RGBA")
        else:
            output_image = Image.fromarray(cleaned_rgb, mode="RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            output_image.save(temp_path)
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def display_image_menu(image_files: list[Path], input_path: Path) -> None:
    """Show a numbered list of all images with their current status."""
    print(f"\n{'=' * 70}")
    print(f"  Images in: {input_path}")
    print(f"{'=' * 70}")

    if not image_files:
        print("  (no images found)")
    else:
        for i, image_path in enumerate(image_files, start=1):
            output_path = output_path_for_image(image_path, input_path)
            if output_path.exists():
                status = "[CLEANED]"
            else:
                status = "[pending]"
            print(f"  {i:4d}. {status} {image_path.name}")

    print(f"{'=' * 70}")
    print(f"  Total: {len(image_files)} image(s)")
    print(f"  Commands: number(s) to clean (e.g., 1,3,5 or 1-3)")
    print(f"            'a' or 'all' to clean all pending images")
    print(f"            'q' or 'quit' to exit")


def parse_selection(user_input: str, image_files: list[Path], input_path: Path) -> list[Path] | None:
    """Parse the user's selection into a list of image paths.

    Returns:
        - list[Path]: the selected image paths
        - empty list: nothing to process (e.g., all already cleaned)
        - None: user wants to quit
    """
    text = user_input.strip().lower()

    if text in {"", "q", "quit", "exit"}:
        return None

    if text in {"a", "all"}:
        pending = [
            image_path
            for image_path in image_files
            if not output_path_for_image(image_path, input_path).exists()
        ]
        if not pending:
            print("All images have already been cleaned.")
        return pending

    selected_indices: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start_idx = int(start_str)
                end_idx = int(end_str)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                for idx in range(start_idx, end_idx + 1):
                    if 1 <= idx <= len(image_files):
                        selected_indices.add(idx)
                    else:
                        print(f"  Warning: {idx} is out of range (1-{len(image_files)})")
            except ValueError:
                print(f"  Warning: invalid range '{part}', ignored")
        else:
            try:
                idx = int(part)
                if 1 <= idx <= len(image_files):
                    selected_indices.add(idx)
                else:
                    print(f"  Warning: {idx} is out of range (1-{len(image_files)})")
            except ValueError:
                print(f"  Warning: invalid input '{part}', ignored")

    if not selected_indices:
        print("  No valid selection.")
        return []

    return [image_files[i - 1] for i in sorted(selected_indices)]


def process_selection(
    selected_images: list[Path],
    input_path: Path,
    *,
    mode: str,
    threshold: int,
    saturation: int,
    min_area: int,
    max_area: int,
    band_lo: int,
    band_hi: int,
    protect_dark: int,
    protect_sat: int,
    dark_dilate: int,
    blackhat: bool,
    blackhat_ksize: int,
    blackhat_thresh: int,
    fill_value: int,
    inpaint: bool,
    inpaint_radius: int,
    preview: bool,
    preview_only: bool,
) -> int:
    """Process the selected images and return the number successfully cleaned."""
    if not selected_images:
        return 0

    failures: list[tuple[Path, Exception]] = []
    for index, image_path in enumerate(selected_images, start=1):
        output_path = output_path_for_image(image_path, input_path)
        print(f"  [{index}/{len(selected_images)}] Cleaning: {image_path.name}")
        try:
            remove_light_gray_watermark(
                image_path,
                output_path,
                mode=mode,
                threshold=threshold,
                saturation=saturation,
                min_area=min_area,
                max_area=max_area,
                band_lo=band_lo,
                band_hi=band_hi,
                protect_dark=protect_dark,
                protect_sat=protect_sat,
                dark_dilate=dark_dilate,
                blackhat=blackhat,
                blackhat_ksize=blackhat_ksize,
                blackhat_thresh=blackhat_thresh,
                fill_value=fill_value,
                inpaint=inpaint,
                inpaint_radius=inpaint_radius,
                preview=preview,
                preview_only=preview_only,
            )
        except Exception as exc:
            failures.append((image_path, exc))
            print(f"    Failed: {exc}")

    cleaned_count = len(selected_images) - len(failures)
    print(f"\n  Cleaned {cleaned_count}/{len(selected_images)} image(s).")

    if failures:
        print("  Failures:")
        for image_path, exc in failures:
            print(f"    - {image_path}: {exc}")

    return cleaned_count


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "threshold",
        "saturation",
        "band_lo",
        "band_hi",
        "protect_dark",
        "protect_sat",
        "blackhat_thresh",
        "fill_value",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 255:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 255.")

    if args.band_lo > args.band_hi:
        raise ValueError("--band-lo must be <= --band-hi.")
    if args.min_area < 0:
        raise ValueError("--min-area must be non-negative.")
    if args.max_area < args.min_area:
        raise ValueError("--max-area must be >= --min-area.")
    if args.dark_dilate < 1:
        raise ValueError("--dark-dilate must be >= 1.")
    if args.blackhat_ksize < 3:
        raise ValueError("--blackhat-ksize must be >= 3.")
    if args.inpaint_radius < 1:
        raise ValueError("--inpaint-radius must be >= 1.")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    input_path = Path(args.path).expanduser().resolve() if args.path else prompt_for_path()
    if not input_path.exists():
        raise FileNotFoundError(f"File or folder not found: {input_path}")

    image_files = collect_image_files(input_path)
    output_root = output_root_for_input(input_path)

    print(f"Cleaned files will be written to: {output_root}")
    print(f"  [mode={args.mode}]")
    if args.mode == "band":
        print(f"  [band={args.band_lo}-{args.band_hi}, sat<={args.saturation}]")
        if args.blackhat:
            print(f"  [blackhat on, ksize={args.blackhat_ksize}, thresh={args.blackhat_thresh}]")
    else:
        print(f"  [bright threshold={args.threshold}, spread<={args.saturation}]")
    if args.inpaint:
        print(f"  [inpainting on, radius={args.inpaint_radius}]")
    else:
        print(f"  [fill-value={args.fill_value}]")
    if args.preview_only:
        print("  [preview-only mode — no cleaning]")

    total_cleaned = 0

    while True:
        display_image_menu(image_files, input_path)

        try:
            user_input = input("\nSelect images to clean: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nExiting. Total images cleaned this session: {total_cleaned}")
            return

        selected = parse_selection(user_input, image_files, input_path)

        if selected is None:
            print(f"Exiting. Total images cleaned this session: {total_cleaned}")
            return

        cleaned = process_selection(
            selected,
            input_path,
            mode=args.mode,
            threshold=args.threshold,
            saturation=args.saturation,
            min_area=args.min_area,
            max_area=args.max_area,
            band_lo=args.band_lo,
            band_hi=args.band_hi,
            protect_dark=args.protect_dark,
            protect_sat=args.protect_sat,
            dark_dilate=args.dark_dilate,
            blackhat=args.blackhat,
            blackhat_ksize=args.blackhat_ksize,
            blackhat_thresh=args.blackhat_thresh,
            fill_value=args.fill_value,
            inpaint=args.inpaint,
            inpaint_radius=args.inpaint_radius,
            preview=args.preview,
            preview_only=args.preview_only,
        )
        total_cleaned += cleaned


if __name__ == "__main__":
    main()
