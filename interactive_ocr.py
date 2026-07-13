"""Interactive OCR entrypoint for Unlimited-OCR.

The script asks for an image, PDF, or folder path, runs Unlimited-OCR, and
writes Markdown result files under the selected output directory.

Version: Accepts image files, PDFs, and batch folder processing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


DEFAULT_REMOTE_MODEL_NAME = "baidu/Unlimited-OCR"
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "models" / "Unlimited-OCR"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"
DEFAULT_PROMPT = (
    "<image>Image parsing"
)

# Noise-only trailing hallucination detection.
# Conservative by design: never treat config-table / CPU / quantity rows as hallucination.
_BROKEN_DET_RE = re.compile(r"<\|det\|>\s*aside_text\b", re.IGNORECASE)
_NON_TEXT_RE = re.compile(r"\[Non-Text\]", re.IGNORECASE)
_DATE_GARBAGE_RE = re.compile(r"^\(?\d*\)?\s*2017年1月1日\s*$")
# Real receipt/table content must never be classified as noise.
_PROTECTED_CONTENT_RE = re.compile(
    r"(?:"
    r"<t[rdh]\b|<table\b|"
    r"海光|CPU|数量|PCS|产品代码|产品编码|配置清单|整机清单|货物明细|"
    r"签收|收货|表格编号|UN-[A-Z0-9-]+|0231[A-Z0-9]+"
    r")",
    re.IGNORECASE,
)


def default_model_name() -> str:
    env_model_path = os.environ.get("UNLIMITED_OCR_MODEL_PATH")
    if env_model_path:
        return str(Path(env_model_path).expanduser())
    if DEFAULT_LOCAL_MODEL_PATH.exists():
        return str(DEFAULT_LOCAL_MODEL_PATH)
    return DEFAULT_REMOTE_MODEL_NAME


def _is_protected_content(text: str) -> bool:
    """True when the line looks like real receipt/table content (never trim/stop on it)."""
    return bool(_PROTECTED_CONTENT_RE.search(text))


def _is_noise_line(line: str) -> bool:
    """Classify a single line as trailing OCR noise (Non-Text / broken aside / date junk)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _is_protected_content(stripped):
        return False
    if _NON_TEXT_RE.search(stripped):
        # Entirely / mostly Non-Text placeholders, or broken det wrapping them.
        remainder = _NON_TEXT_RE.sub("", stripped)
        remainder = re.sub(r"<\|/?det\|>", "", remainder, flags=re.IGNORECASE)
        remainder = re.sub(r"aside_text\b", "", remainder, flags=re.IGNORECASE)
        remainder = re.sub(r"[\[\](),.\d\s]+", "", remainder)
        return len(remainder) <= 8
    if _DATE_GARBAGE_RE.match(stripped):
        return True
    if _BROKEN_DET_RE.match(stripped) and "<|/det|>" not in stripped:
        return True
    return False


def _trailing_noise_run_start(lines: list[str]) -> int | None:
    """Return the line index where a trailing all-noise run begins, or None."""
    index = len(lines) - 1
    # Ignore a single final empty line when locating the run.
    if index >= 0 and not lines[index].strip():
        index -= 1

    run_end = index
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            index -= 1
            continue
        if not _is_noise_line(lines[index]):
            break
        index -= 1

    run_start = index + 1
    # Skip leading blanks inside the candidate run so counting is on noise lines only.
    while run_start <= run_end and not lines[run_start].strip():
        run_start += 1

    noise_count = sum(1 for line in lines[run_start : run_end + 1] if line.strip())
    if noise_count <= 0:
        return None
    return run_start if noise_count else None


def trailing_hallucination_span(text: str, min_repeats: int = 4) -> tuple[int, int] | None:
    """If a trailing noise loop is found, return (cut_from, end) covering the whole noise tail.

    Only triggers on noise (Non-Text / broken aside_text / known junk dates).
    Does not use digit-normalized structural matching, so similar config-table rows
    with different CPU/quantity fields are never treated as hallucination.
    """
    if min_repeats < 2 or not text:
        return None

    lines = text.splitlines(keepends=True)
    if not lines:
        return None

    run_start = _trailing_noise_run_start(lines)
    if run_start is None:
        return None

    noise_count = sum(1 for line in lines[run_start:] if line.strip() and _is_noise_line(line))
    if noise_count < min_repeats:
        return None

    # Extra safety: never cut if the proposed cut point sits inside protected table markup.
    prefix = "".join(lines[:run_start])
    if prefix.rstrip().endswith(("<tr>", "<td>", "<th>", "<table>")):
        return None

    cut_from = sum(len(line) for line in lines[:run_start])
    return cut_from, len(text)


def has_trailing_hallucination(text: str, min_repeats: int = 4) -> bool:
    return trailing_hallucination_span(text, min_repeats=min_repeats) is not None


def trim_trailing_hallucination(text: str, min_repeats: int = 4) -> str:
    """Remove the entire trailing noise run; leave all prior receipt/table content intact."""
    span = trailing_hallucination_span(text, min_repeats=min_repeats)
    if span is None:
        return text
    cut_from, _ = span
    trimmed = text[:cut_from].rstrip()
    if not trimmed:
        return text
    return trimmed + ("\n" if text.endswith("\n") else "")


def create_trailing_hallucination_stopping_criteria(
    tokenizer,
    prompt_length: int,
    min_repeats: int = 4,
    check_every: int = 16,
):
    """Build a transformers StoppingCriteria that stops on trailing noise loops only."""
    import torch
    from transformers import StoppingCriteria

    class TrailingHallucinationStoppingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.min_repeats = min_repeats
            self.check_every = max(1, check_every)
            self.steps = 0
            self.triggered = False

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
            self.steps += 1
            batch_size = input_ids.shape[0]
            device = input_ids.device
            should_stop = torch.zeros(batch_size, dtype=torch.bool, device=device)

            if self.steps % self.check_every != 0:
                return should_stop

            for batch_index in range(batch_size):
                generated = input_ids[batch_index, self.prompt_length :]
                if generated.numel() == 0:
                    continue
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                if has_trailing_hallucination(text, min_repeats=self.min_repeats):
                    should_stop[batch_index] = True
                    self.triggered = True

            if bool(should_stop.any()):
                print(
                    f"\n[early-stop] Trailing noise hallucination detected "
                    f"(≥{self.min_repeats} Non-Text/aside junk lines); stopping generation.",
                    flush=True,
                )
            return should_stop

    return TrailingHallucinationStoppingCriteria()


@contextmanager
def hallucination_early_stop(
    model,
    tokenizer,
    enabled: bool = True,
    min_repeats: int = 4,
    check_every: int = 16,
):
    """Inject streaming early-stop into model.generate for the duration of OCR."""
    if not enabled:
        yield None
        return

    from transformers import StoppingCriteriaList

    original_generate = model.generate
    criteria_holder: dict[str, object] = {"criteria": None}

    def generate_with_early_stop(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return original_generate(*args, **kwargs)

        prompt_length = int(input_ids.shape[-1])
        stop_criteria = create_trailing_hallucination_stopping_criteria(
            tokenizer,
            prompt_length=prompt_length,
            min_repeats=min_repeats,
            check_every=check_every,
        )
        criteria_holder["criteria"] = stop_criteria

        existing = kwargs.get("stopping_criteria")
        if existing is None:
            kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])
        elif isinstance(existing, StoppingCriteriaList):
            existing.append(stop_criteria)
            kwargs["stopping_criteria"] = existing
        else:
            kwargs["stopping_criteria"] = StoppingCriteriaList([*list(existing), stop_criteria])

        return original_generate(*args, **kwargs)

    model.generate = generate_with_early_stop
    try:
        yield criteria_holder
    finally:
        model.generate = original_generate


def apply_hallucination_trim(markdown_path: Path, min_repeats: int = 4) -> bool:
    """Trim leftover trailing noise from result.md after early-stop or full decode."""
    if not markdown_path.exists():
        return False
    original = markdown_path.read_text(encoding="utf-8")
    trimmed = trim_trailing_hallucination(original, min_repeats=min_repeats)
    if trimmed == original:
        return False
    markdown_path.write_text(trimmed, encoding="utf-8")
    print(f"[early-stop] Trimmed trailing hallucination from {markdown_path}", flush=True)
    return True


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
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable streaming early-stop for trailing Non-Text/aside noise loops.",
    )
    parser.add_argument(
        "--hallucination-repeats",
        type=int,
        default=4,
        help=(
            "Stop after this many consecutive trailing noise lines "
            "([Non-Text] / broken aside_text / junk dates). "
            "Does not match config-table rows. Default: 4."
        ),
    )
    parser.add_argument(
        "--hallucination-check-every",
        type=int,
        default=16,
        help="Check for trailing noise hallucination every N newly generated tokens. Default: 16.",
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
    min_repeats = max(2, args.hallucination_repeats)
    with hallucination_early_stop(
        model,
        tokenizer,
        enabled=not args.no_early_stop,
        min_repeats=min_repeats,
        check_every=args.hallucination_check_every,
    ):
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

    if isinstance(outputs, str):
        outputs = trim_trailing_hallucination(outputs, min_repeats=min_repeats)

    markdown_path = write_fallback_markdown(output_dir, outputs)
    apply_hallucination_trim(markdown_path, min_repeats=min_repeats)
    return markdown_path


# Convert one PDF to images and parse each page with single-image OCR (crop mode).
def run_pdf_ocr(model, tokenizer, pdf_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_pdf_") as temp_root:
        temp_root_path = Path(temp_root)
        image_paths = pdf_to_images(pdf_path, temp_root_path / "pages", args.dpi)
        print(f"Parsing PDF with {len(image_paths)} page(s) via single-image mode")

        page_markdowns: list[str] = []
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for page_index, image_path in enumerate(image_paths):
            page_dir = temp_root_path / f"out_{page_index}"
            page_dir.mkdir()
            print(f"  page {page_index + 1}/{len(image_paths)}")
            run_image_ocr(model, tokenizer, Path(image_path), page_dir, args)

            markdown = (page_dir / "result.md").read_text(encoding="utf-8")
            page_images = page_dir / "images"
            if page_images.exists():
                for image_file in sorted(page_images.iterdir()):
                    if not image_file.is_file():
                        continue
                    dest_name = f"page_{page_index}_{image_file.name}"
                    shutil.copy2(image_file, images_dir / dest_name)
                    markdown = markdown.replace(
                        f"![](images/{image_file.name}",
                        f"![](images/{dest_name}",
                    )
            page_markdowns.append(markdown.strip())

            boxes = page_dir / "result_with_boxes.jpg"
            if boxes.exists():
                shutil.copy2(boxes, output_dir / f"result_with_boxes_{page_index}.jpg")

    markdown_path = output_dir / "result.md"
    markdown_path.write_text("<PAGE>\n" + "\n<PAGE>\n".join(page_markdowns) + "\n", encoding="utf-8")
    return markdown_path


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
