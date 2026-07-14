from pathlib import Path

from paddleocr import PaddleOCRVL
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/home/lei/OCR/output")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"


def prompt_for_path() -> Path:
    while True:
        value = input("请输入文件或文件夹路径: ").strip().strip('"').strip("'")
        if not value:
            print("路径不能为空，请重新输入。")
            continue

        path = Path(value).expanduser().resolve()
        if path.exists() and (path.is_file() or path.is_dir()):
            return path

        print(f"文件或文件夹不存在: {path}")


def is_supported_file(path: Path) -> bool:
    if path.name.startswith("._"):
        return False
    suffix = path.suffix.lower()
    return suffix == PDF_SUFFIX or suffix in IMAGE_SUFFIXES


def collect_supported_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and is_supported_file(path)
    )


def resolve_save_dir(input_path: Path, source_file: Path) -> Path:
    """Single file -> OUTPUT_ROOT; folder -> OUTPUT_ROOT/<folder_name>/..."""
    if input_path.is_file():
        return OUTPUT_ROOT

    rel_parent = source_file.relative_to(input_path).parent
    save_dir = OUTPUT_ROOT / input_path.name / rel_parent
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(
            PROJECT_ROOT / "models/PaddleOCR-VL-1.6/PP-DocLayoutV3"
        ),
        vl_rec_model_dir=str(
            PROJECT_ROOT / "models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-0.9B"
        ),
        device="gpu:0",
    )

    input_path = prompt_for_path()

    if input_path.is_file():
        if not is_supported_file(input_path):
            raise SystemExit(f"不支持的文件类型: {input_path}")
        files = [input_path]
        print(f"将处理 1 个文件，Markdown 保存到: {OUTPUT_ROOT}")
    else:
        files = collect_supported_files(input_path)
        if not files:
            raise SystemExit(f"文件夹中没有可处理的图片或 PDF: {input_path}")
        folder_out = OUTPUT_ROOT / input_path.name
        folder_out.mkdir(parents=True, exist_ok=True)
        print(
            f"将处理 {len(files)} 个文件，Markdown 保存到文件夹: {folder_out}"
        )

    results = pipeline.predict([str(path) for path in files])
    for res in tqdm(results, total=len(files), desc="OCR"):
        source = Path(res["input_path"]).resolve()
        save_dir = resolve_save_dir(input_path, source)
        res.save_to_markdown(save_path=save_dir)
        print(f"已保存: {res.save_path}")


if __name__ == "__main__":
    main()
