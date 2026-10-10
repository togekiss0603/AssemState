# Paper, website, and prompt correspondence

Paper: [AssemState: Manual and Physical-State-Guided Reasoning for Zero-shot Furniture Assembly](https://arxiv.org/abs/2610.08446), Zhiyuan Qi, Jierui Li, Yifan Shen, Cheng Qian, and Jiateng Liu (2026).

This audit uses the author-supplied manuscript source archive dated October 10, 2026. It compares that archive with the public release; it does not claim that every byte of the arXiv PDF is identical to the supplied source. No model inference or new physics experiment was run for this audit.

## Website coverage

The [project page](https://assemstate.wirymouse1.chatgpt.site) now contains all six table environments as seven HTML table panels. All rows and metric columns are included. Appendix material is visible by expanding the Appendix sections.

| Manuscript label | Website anchor | Content |
|---|---|---|
| `tab:tree_generation` | `#table-tree` | All six methods; P/R/F1/Tree EM |
| `tab:pose_sim_result` (a) | `#table-pose` | Both backbones; judge, RMSE, GD, mean/median CD |
| `tab:pose_sim_result` (b) | `#table-release` | Both backbones; stability, drift, Z drop |
| `tab:pose_ablation` | `#table-ablation` | All three component settings |
| `tab:manual_stage1_example` | `#table-semantic` | Retained four-part bed semantics |
| `tab:pose_trace` | `#table-trace` | Three rejected Bench/applaro rounds |
| `tab:pose_trace_config` | `#table-config` | Proposer, selector, and Judge settings |

All seven active manuscript figures are shown: overview, manual-to-operation reasoning, pose refinement, failure examples, hierarchical action complexity, release-threshold sensitivity, and reported failure labels. Source and rendered-image hashes are recorded in [paper-figure provenance](../website/assets/paper/PROVENANCE.json). Unused figures in the manuscript ZIP are not treated as missing paper content.

The page condenses prose and links the full paper and full available prompt snapshots. It is not a verbatim web transcription of every equation, paragraph, reference, or ethical statement.

## Exact prompt mapping

Paths below are relative to the repository. The seven raw prompts in `artifacts/paper_prompts/` match the manuscript's `appendix_materials/raw/` byte for byte. The ten files in `artifacts/appendix_print/` are exact copies of the manuscript's typeset-prompt source files, including its schedule and excerpts.

| Appendix prompt | Raw file / tree runtime filename | Comparison with appendix print |
|---|---|---|
| M1 `part_names_print.txt` | `generate_json.txt` | Exact bytes |
| M2 `part_descriptions_print.txt` | `select_material.txt` | Exact bytes |
| M3 `tree_v2m_print.txt` | `planning_anchor_v2m.txt` | Only `—` → `--` and `✓` → `[OK]` |
| M4 `tree_conversion_print.txt` | `tree_ikea_manual.txt` | Only curly apostrophe `’` → `'` |
| `pose_initial_print.txt` | `pose_initial_prompt.txt` | Exact bytes |
| `pose_feedback_print.txt` | `pose_feedback_prompt.txt` | Exact bytes |
| `pose_selector_print.txt` | `pose_selector_prompt.txt` | Only `≈` → `~` and `–` → `-` |

All four M1–M4 files in `tree/vlm_planner/prompts/` exactly match the raw manuscript prompts. `scripts/run_tree.py` selects the v2m path in `run_anchor.py`; its M3 template is `planning_anchor_v2m.txt`. The three phases inside M3 are one model request, not three independent calls. M4 is the text-only fallback after rule-based parsing. Prompt bytes have not been rewritten to match typographical substitutions in the appendix.

The historical M3 word “ROOT” refers to its final-step anchor. This differs from the graph-theoretic root of the complete assembly, as the appendix already explains. The snapshot is preserved rather than silently corrected.

## Pose correspondence is archival, not executable equivalence

The three pose prompts are real retained requests for the historical Bench/applaro trace. They are **not** the templates called by the current maintained agent. The current `pose_estimation/visual_agent/prompts.py` builds evaluator and proposer prompts through `build_evaluator_prompt` and `build_proposer_prompt`, with a different selection and iteration schedule. Setting three iterations does not recreate the historical proposer → render → selector → Judge loop.

The complete historical Judge request, support-view-selection request, system/API wrapper, and original experiment runner were not recovered. The appendix reports the saved Judge response schema and diagnoses; these must not be presented as a recovered Judge prompt. The representative trace uses gpt-5.2 and one candidate per round, despite aggregate table labels saying GPT-5 and some historical filenames containing `k3`.

Table A1 is a bed example; Table A2 is the separate Bench/applaro failure trace. The retained bed example has no final tree output, and the bench trace has no accepted round. Neither is represented as a successful end-to-end assembly on the website.

## Checks and remaining limits

```bash
python scripts/audit_paper_alignment.py
python scripts/audit_paper_alignment.py --paper-zip /path/to/manuscript.zip
python scripts/reproduce_tables.py
```

The first command verifies prompt correspondence, the seven website table panels, local links, and the seven figure hashes using only released files. The optional ZIP check additionally verifies original prompt bytes, source figures, all ten print snapshots, and the complete manuscript table-label inventory. Table values were transcribed and checked against `latex/sec/4_exp.tex` and `latex/sec/appendix.tex`; the script's coverage check alone does not certify all table numbers. The existing table-reproduction command independently recomputes archived pose/release aggregates from 1,701 records.

The unresolved tree P/R/F1 aggregation discrepancy, incomplete original tree predictions, nonidentical operation contexts across some configurations, and historical/current-agent differences remain documented in [PROTOCOL.md](PROTOCOL.md). This update improves correspondence and presentation; it does not supply missing experiments or claim full historical reproduction.
