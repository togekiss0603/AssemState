# Evaluation and reproduction protocol

## Structure recovery

The planner observes a numbered scene and available annotated views. It first associates part IDs with names using the manual preview, then enriches the descriptions using the full rendered manual-page sequence. Anchor-v2m receives the semantic table, scene, and ordered numbered step masks. All anchor/visibility/subassembly phases happen inside one model request. A rule parser or text-only conversion call produces a nested-list tree.

Inputs include precomputed masks; this repository does not claim raw-PDF segmentation performance. Leaves identify rigid parts; tree structure is not a robot trajectory. The original evaluator ignores sibling order and can permute annotated equivalent parts when `parts_ct < 9`. Tree JSON may be a JSON-encoded string containing a list; the loader supports both formats.

## Historical current-operation pose experiment

There are 55 designated correct-tree products and 243 operations per setting. Each operation uses a correct moving/fixed-part specification and correct preceding assembly state. Target poses are withheld from the model. Predictions are independently scored before a ground-truth reset constructs the next operation's context; later states do not validate a successful predicted rollout.

The preserved three-round trace contains proposer, candidate rendering, visual selector and saved Judge diagnoses. Filenames containing `k3` do not establish three candidates; the representative preserved trace has K=1. Prompt snapshots in `artifacts/paper_prompts/` came directly from the user-supplied manuscript. They are not a replacement for the missing executable historical harness or missing full Judge/support-view-selection prompts.

Archived summaries provide RMSE_t, geodesic rotation distance, bidirectional squared Chamfer components, PA, acceptance, and release measurements. `reproduce_tables.py` recomputes aggregates from those per-operation values. It does not recompute point-cloud metrics from raw predictions. Units are evaluator/dataset native, not calibrated real furniture dimensions.

## Historical release diagnostic

The recorded release implementation is `isaaclab_release_aabb.v1`: fixed parts use static axis-aligned boxes and only the moving part is dynamic. Stability requires translation drift < 0.05, vertical drop < 0.05, and rotation drift < 12 degrees. This is a coarse physical diagnostic, not connector-level contact simulation or a robot insertion test. Simulator settings are preserved inside each release JSON.

The maintained agent's offline renderer and optional Isaac Sim backend do not reproduce this historical release experiment. No simulator rerun is claimed by the table script.

## Known discrepancies

- The supplied manuscript reports tree Precision 62.34, Recall 62.46 and F1 62.80. Under common averaging on the same samples/weights, F1 cannot exceed the arithmetic mean of P/R (62.40). Full original predictions and aggregation details are required to resolve this. The release does not silently alter or certify these values.
- Complete historical v2m and without-anchor prediction sets were not recovered. Legacy cached examples cannot substantiate the paper's tree table and are not bundled in the public release.
- GPT structured vs self-check shares only 160/243 exact `(product, operation index, moving part, fixed set)` contexts. GPT multi-view vs self-check and Haiku structured vs self-check share all 243. Matching keys does not guarantee identical initial images or prompts.
- The manuscript's table label is GPT-5; the available representative historical trace uses `gpt-5.2`. This release retains both descriptions rather than claiming all historical requests were independently verified.
- `Chair/fanbyn` has four numbered masks for five metadata steps, and `Table/sjalland` has five for seven. The existing planner reads the available files. The data validator exposes this coverage discrepancy; it does not synthesize inputs.

## Maintained agent

The included agent uses evaluator → proposer → update, optional multi-path sampling and programmatic selection. The default target-free mode avoids loading target meshes. Oracle/debug modes remain available and must not be compared as target-free predictions. The agent stops on failed operations by default and does not implement the archived GT-reset experiment. The maintained operation ordering can differ from the representative paper trace.

Live model calls and optional Isaac Sim execution have not been exercised during release preparation. Offline CPU rendering, no-key behavior, target-mesh exclusion, archived record integrity, and clone portability are the release checks.
