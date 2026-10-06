"""Import the supplied source snapshot without weights, datasets or nested copies."""
import hashlib
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {
    "__init__.py", "make_polyhedron_objs.py", "generate_yolo_coco_composite.py",
    "run_yolo_parallel.py", "run_yolo_coco_composite_batch.py",
    "export_meta_v2_model_datasets.py", "export_meta_v2_cube_face_unified_dataset.py",
    "download_coco2017_backgrounds.py", "download_generic_fruit_textures.py",
    "prepare_fruit_textures.py", "prepare_kaggle_fruits360_textures.py",
    "prepare_fruitseg30_textures.py", "make_mixed_fruit_textures.py",
    "audit_fruit_generation.py", "make_texture_contact_sheets.py",
}


def main():
    parser = argparse.ArgumentParser(description="Archive a supplied Data_Generation_Blender source snapshot.")
    parser.add_argument("source", type=Path, help="Path to the extracted upstream snapshot")
    args = parser.parse_args()
    source_root = args.source.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source snapshot directory does not exist: {source_root}")
    destination = ROOT / "perception" / "generation"
    records = []
    for folder in ("scripts", "jetson", "docs", "reports", "assets/generated"):
        for source_file in sorted((source_root / folder).rglob("*")):
            if not source_file.is_file() or source_file.suffix.lower() not in {
                ".py", ".ps1", ".sh", ".bat", ".md", ".json", ".csv", ".obj"
            }:
                continue
            if source_file.stat().st_size > 2_000_000:
                continue
            rel = source_file.relative_to(source_root)
            archived = destination / "history" / "upstream" / rel
            archived.parent.mkdir(parents=True, exist_ok=True)
            source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            if archived.exists() and hashlib.sha256(archived.read_bytes()).hexdigest() != source_hash:
                raise RuntimeError(f"Refusing to overwrite archived history: {archived}")
            if not archived.exists():
                shutil.copy2(source_file, archived)
            records.append({"source": rel.as_posix(), "sha256": source_hash})
            if folder == "scripts" and source_file.name in ACTIVE:
                target = destination / "scripts" / source_file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
                    print(f"Keeping locally adapted code: {target}")
                if not target.exists():
                    shutil.copy2(source_file, target)
            elif folder == "assets/generated":
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source_file, target)
    for name in ("readme.md", "readme_specific.md", "HISTORY.md", "EXPERIMENTS.md", "BACKUP_MANIFEST.md", "environment.yml", "main.py"):
        source_file = source_root / name
        target = destination / "history" / "upstream" / name
        source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError(f"Refusing to overwrite archived history: {target}")
        if not target.exists():
            shutil.copy2(source_file, target)
        records.append({"source": name, "sha256": source_hash})
    manifest_path = destination / "history" / "import-manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    (destination / "history" / "import-manifest.json").write_text(json.dumps({
        "upstream": "https://github.com/jaeyoungi2006/Data_Generation_Blender",
        "snapshot": "User-supplied ZIP directory; upstream commit not available",
        "excluded": ["nested duplicate directory", "datasets", "weights", "binary reports", "private SMTP configuration"],
        "integration_changes": existing_manifest.get("integration_changes", []),
        "files": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Archived {len(records)} files; active scripts: {len(ACTIVE)}")


if __name__ == "__main__":
    main()
