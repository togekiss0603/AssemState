# External data and illustrative sample

The full dataset and furniture meshes are not distributed in this public repository. Five user-authorized Bench/applaro input images are included under `website/assets/` solely for illustration, with source paths and hashes in `PROVENANCE.json`. They are not a complete runnable data fixture.

## Expected input structure

```text
assemblydata/
  main_data.json
  prescene/<category>/<name>/
    scene_annotated.png
    scene_rot_annotated.png
    scene_*_annotated.png
    scene.obj
    scene_parts/*.obj
  parts/<category>/<name>/*.obj
  pdfs/<category>/<name>/page_*.png
  mask/<category>/<name>/*mask_overlay_cropped_numbered.png
  preassembly_scenes/<category>/<name>/equiv_parts.txt  # when present
```

The source metadata contain 102 products, 754 rigid parts and 393 annotated manual steps. Obtain the original assets under their applicable terms; this repository provides no public full-dataset download. Manual pages are pre-rendered PNG files, not PDFs parsed during inference.

Prepare a runtime subset from an existing authorized data copy:

```bash
python scripts/prepare_data.py --source /path/to/original/assemblydata \
  --destination data/assemblydata
python scripts/prepare_data.py --source data/assemblydata --check
python scripts/run_tree.py --data-root data/assemblydata --all
```

The destination must be new. The preparer copies only runtime inputs, writes file checksums and excludes logs, credentials, training checkpoints and projection pickles. `--archive /path/to/asset.tar.gz` can instead package an authorized local copy; packaging does not change data rights.

`--strict-step-coverage` makes validation fail if numbered masks are missing for metadata steps. In the inspected original data, `Chair/fanbyn` has 4 masks for 5 steps and `Table/sjalland` has 5 for 7. The code reports these differences rather than fabricating masks.

## Bench/applaro inputs

| Source-relative file | Consumer |
|---|---|
| `prescene/Bench/applaro/scene_annotated.png` | Semantic association and default v2m planning |
| `prescene/Bench/applaro/scene_rot_annotated.png` | Available rotated observation for semantic association |
| `pdfs/Bench/applaro/page_1.png` | Manual preview in the first semantic call; full pages in the second |
| `mask/Bench/applaro/step_0_mask_overlay_cropped_numbered.png` | First numbered mask in v2m planning |
| `mask/Bench/applaro/step_1_mask_overlay_cropped_numbered.png` | Second numbered mask in v2m planning |

The reference metadata contain four parts, two steps and tree `[[0,1,2],3]`. Both steps refer to manual page 4. This is a reference annotation, not an AssemState prediction. Source image pixels are copied unchanged for the project page.

## Maintained pose inputs

In addition to meshes and image assets, provide a semantic-output root containing `<category>/<name>/stage1_output.json` and `stage2_output.txt` (or their `debug/` variants). A `tree.json` or `pose_atomic_contract.json` may define operation grouping. The tree runner's `outputs/tree/anchor_v2m` directory can be used as that root.

Pass `--data-root` and `--manual2skill-output` explicitly. Target-free mode does not load target meshes for model observation; target assets remain required by the inherited input validator. Test optional local fixtures by placing a complete Bench/applaro data subset under `examples/data` and corresponding semantic outputs under `examples/stage_outputs`; these paths are ignored by Git in the public release.
