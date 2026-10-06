# Perception data generation

The supported clean-checkout path is [`../docs/synthetic-data-reproduction.md`](../docs/synthetic-data-reproduction.md). Run the commands there from this directory. They download a checksum-verified public Fruits-360 subset, generate arena scenes and Meta V2 visibility labels, create a shared scene split, export the A1 and face datasets, and train the models.

## Active scripts

- `prepare_assets.py` builds the pinned public fruit texture pack and records file hashes and attribution.
- `scripts/make_polyhedron_objs.py` creates the 8 cm shape meshes.
- `scripts/run_yolo_parallel.py` launches seeded BlenderProc workers and resumes complete image/label/metadata triples.
- `scripts/generate_yolo_coco_composite.py` renders the scene, final camera pixels, segmentation labels and Meta V2 metadata. The integrated version uses the final arena print-face rule (top plus opposing Y sides).
- `scripts/split_meta_v2_dataset.py` assigns whole scenes to train, validation and test with a fixed seed.
- `scripts/export_meta_v2_model_datasets.py` exports A1 labels and classifier crops with the shared scene split.
- `scripts/export_meta_v2_cube_face_unified_dataset.py` exports 224 px face-segmentation crops with the same split.

The active entry points have their dependencies in `requirements.txt`. GPU-enabled PyTorch is installed separately for the machine's CUDA runtime; the supported version pair and commands are in the reproduction guide.

## Historical source and experiments

`history/upstream/` preserves the supplied `Data_Generation_Blender` snapshot, source documentation, scripts and text reports. Binary datasets, model weights, nested duplicate checkouts and private credentials were excluded. `history/import-manifest.json` records the source path and SHA256 for each imported file. The supplied snapshot did not contain a commit id.

The public recipe intentionally differs from the historical competition dataset: it uses a pinned public texture subset and procedural arena backgrounds, and it follows the final competition face placement rule. The experiment outcomes and non-redistributed dataset inventory are documented in [`../docs/synthetic-data-experiments.md`](../docs/synthetic-data-experiments.md).
