"""Interactive OCR entrypoint for Unlimited-OCR.

The script asks for an image, PDF, or folder path, runs Unlimited-OCR, and
writes Markdown result files under the selected output directory.

Version: Accepts image files, PDFs, and batch folder processing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


DEFAULT_REMOTE_MODEL_NAME = "baidu/Unlimited-OCR"
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "models" / "Unlimited-OCR"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"
DEFAULT_PROMPT = (
    "<image>OCR this delivery document exactly. Preserve table structure. "
    "Only output text visibly present in the image. Do not infer, continue, "
    "or repeat table rows. If a cell is unclear, leave it blank. Stop after "
    "the last visible content on the page."
)


def default_model_name() -> str:
    env_model_path = os.environ.get("UNLIMITED_OCR_MODEL_PATH")
    if env_model_path:
        return str(Path(env_model_path).expanduser())
    if DEFAULT_LOCAL_MODEL_PATH.exists():
        return str(DEFAULT_LOCAL_MODEL_PATH)
    return DEFAULT_REMOTE_MODEL_NAME


# Parse command-line options while keeping interactive prompts as the default flow.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unlimited-OCR on an image, PDF, or folder.")
    parser.add_argument(
        "--file",
        "-f",
        help="Path to an image, PDF, or folder. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directory where per-file result.md and result_with_boxes.jpg outputs will be written.",
    )
    parser.add_argument("--model", default=default_model_name(), help="Model name or local model path.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when converting PDF pages.")
    parser.add_argument("--max-length", type=int, default=32768, help="Maximum generation length.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="OCR prompt. Must include one <image> token.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 uses deterministic decoding.",
    )
    return parser.parse_args()


# Ask the user for an input path and accept either a single file or a folder.
def prompt_for_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip().strip('"').strip("'")
        if not value:
            print("Please enter a path.")
            continue

        path = Path(value).expanduser().resolve()
        if path.exists() and (path.is_file() or path.is_dir()):
            return path

        print(f"File or folder not found: {path}")


# Ask the user where OCR outputs should be stored, with a stable default path.
def prompt_for_output_dir(input_path: Path) -> Path:
    default_dir = Path.cwd() / "ocr_outputs" / input_path.stem
    value = input(f"Output directory [{default_dir}]: ").strip().strip('"').strip("'")
    return Path(value).expanduser().resolve() if value else default_dir


# Check whether a single file has an extension supported by this script.
def is_supported_file(path: Path) -> bool:
    if path.name.startswith("._"):
        return False

    suffix = path.suffix.lower()
    return suffix == PDF_SUFFIX or suffix in IMAGE_SUFFIXES


# Find every supported image or PDF in a folder, recursively and in a stable order.
def collect_supported_files(folder_path: Path) -> list[Path]:
    return sorted(path for path in folder_path.rglob("*") if path.is_file() and is_supported_file(path))


# Convert a PDF into temporary page images so Unlimited-OCR can parse it as pages.
def pdf_to_images(pdf_path: Path, temp_dir: Path, dpi: int) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF. Install it with: pip install pymupdf") from exc

    image_paths: list[str] = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for page_index, page in enumerate(doc, start=1):
            image_path = temp_dir / f"page_{page_index:04d}.png"
            page.get_pixmap(matrix=matrix).save(str(image_path))
            image_paths.append(str(image_path))
    finally:
        doc.close()

    if not image_paths:
        raise RuntimeError(f"No pages were found in PDF: {pdf_path}")

    return image_paths


# Load the tokenizer and model once so single-file and folder runs share the same instance.
def load_model(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Unlimited-OCR inference requires an NVIDIA GPU with CUDA available.")

    local_model_path = Path(model_name).expanduser()
    is_local_model = local_model_path.exists()
    resolved_model_name = str(local_model_path.resolve()) if is_local_model else model_name
    load_kwargs = {"trust_remote_code": True, "local_files_only": is_local_model}

    print(f"Loading model: {resolved_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, **load_kwargs)
    model = AutoModel.from_pretrained(
        resolved_model_name,
        **load_kwargs,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    return tokenizer, model


def delete_path(output_path: Path) -> None:
    if output_path.is_dir() and not output_path.is_symlink():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


# Remove stale files before OCR, and prune extra files after OCR when keep is provided.
def clean_output_dir(output_dir: Path, keep: set[str] | None = None) -> None:
    if not output_dir.exists():
        return

    for output_path in output_dir.iterdir():
        if keep and output_path.name in keep and output_path.is_file():
            continue

        delete_path(output_path)


# Ensure a Markdown result exists, using returned model text only if the model did not save one.
def write_fallback_markdown(output_dir: Path, outputs: str | None) -> Path:
    markdown_path = output_dir / "result.md"
    if markdown_path.exists():
        return markdown_path

    if outputs is None:
        raise RuntimeError(f"OCR finished, but no Markdown file was produced in {output_dir}")

    markdown_path.write_text(outputs, encoding="utf-8")
    return markdown_path


# Run the single-image inference path recommended by the Unlimited-OCR README.
def run_image_ocr(model, tokenizer, image_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    outputs = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        max_length=args.max_length,
        no_repeat_ngram_size=35,
        ngram_window=128,
        temperature=args.temperature,
        save_results=True,
    )
    return write_fallback_markdown(output_dir, outputs)


# Convert one PDF to images and parse all pages through the model's multi-page path.
def run_pdf_ocr(model, tokenizer, pdf_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_pdf_") as temp_root:
        image_paths = pdf_to_images(pdf_path, Path(temp_root) / "pages", args.dpi)
        print(f"Parsing PDF with {len(image_paths)} page(s)")
        outputs, _ = model.infer_multi(
            tokenizer,
            prompt=args.prompt,
            image_files=image_paths,
            output_path=str(output_dir),
            image_size=1024,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            temperature=args.temperature,
            save_results=True,
        )

    return write_fallback_markdown(output_dir, outputs)


# Process one supported file and return the path to its Markdown output.
def process_file(model, tokenizer, input_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)

    if input_path.suffix.lower() == PDF_SUFFIX:
        markdown_path = run_pdf_ocr(model, tokenizer, input_path, output_dir, args)
    else:
        markdown_path = run_image_ocr(model, tokenizer, input_path, output_dir, args)

    return markdown_path


# Build a per-file output directory for folder inputs while preserving nested folders.
def output_dir_for_batch_file(input_file: Path, input_folder: Path, batch_output_dir: Path) -> Path:
    relative_path = input_file.relative_to(input_folder)
    return batch_output_dir / relative_path.with_suffix("")


# Process all supported images and PDFs found under a folder.
def process_folder(model, tokenizer, folder_path: Path, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    input_files = collect_supported_files(folder_path)
    if not input_files:
        supported = ", ".join(sorted(IMAGE_SUFFIXES | {PDF_SUFFIX}))
        raise ValueError(f"No supported files found in {folder_path}. Supported types: {supported}")

    markdown_paths: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    print(f"Found {len(input_files)} supported file(s) in: {folder_path}")

    for index, input_file in enumerate(input_files, start=1):
        file_output_dir = output_dir_for_batch_file(input_file, folder_path, output_dir)
        print(f"\n[{index}/{len(input_files)}] Parsing: {input_file}")
        try:
            markdown_paths.append(process_file(model, tokenizer, input_file, file_output_dir, args))
        except Exception as exc:
            failures.append((input_file, exc))
            print(f"Failed to parse {input_file}: {exc}")

    if failures:
        print("\nSome files failed:")
        for input_file, exc in failures:
            print(f"- {input_file}: {exc}")

    return markdown_paths


# Coordinate input validation, model loading, and file or folder OCR execution.
def main() -> None:
    args = parse_args()
    input_path = Path(args.file).expanduser().resolve() if args.file else prompt_for_path("Image, PDF, or folder path: ")
    if not input_path.exists():
        raise FileNotFoundError(f"File or folder not found: {input_path}")
    if "<image>" not in args.prompt:
        raise ValueError("The OCR prompt must include one <image> token.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else prompt_for_output_dir(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model)
    if input_path.is_dir():
        markdown_paths = process_folder(model, tokenizer, input_path, output_dir, args)
        print(f"\nFinished. Markdown files written: {len(markdown_paths)}")
        for markdown_path in markdown_paths:
            print(f"- {markdown_path}")
    else:
        if not is_supported_file(input_path):
            supported = ", ".join(sorted(IMAGE_SUFFIXES | {PDF_SUFFIX}))
            raise ValueError(f"Unsupported file type '{input_path.suffix}'. Supported types: {supported}")
        named_markdown = process_file(model, tokenizer, input_path, output_dir, args)
        print(f"\nMarkdown written to: {named_markdown}")


if __name__ == "__main__":
    main()
