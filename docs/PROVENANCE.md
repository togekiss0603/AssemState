# Public release provenance

Prepared from the user-confirmed AssemState clean handoff and supplied manuscript on 2026-10-10. The later MANUAL–STATE training handoff is excluded.

- Core tree and maintained pose code: June clean handoff, with September handoff notes.
- Tree protocol: September 10 v2m export; the four distributed tree prompts match the manuscript copies byte for byte.
- Archived numeric pose/release records: the seven runs identified by the manuscript audit. Machine-local summary paths were omitted; numbers were retained.
- Paper prompts and illustrative responses: supplied manuscript ZIP, `appendix_materials/raw`.
- Website sample: five explicitly authorized Bench/applaro images from the source dataset; see `website/assets/PROVENANCE.json`.

The public repository starts with a clean history. It does not inherit the private release's complete example assets, meshes, data archive, old credentials or work-directory history. A separate local preparation package exists but is not uploaded.

Release changes replace hardcoded paths, restrict tree inference to the released v2m prompts, remove provider-specific credential aliases and default relay URLs, and require explicit `--dry-run` for mock inference. The table audit, pinned dependencies, data preparation, integrity check, tests and website are release additions.

Original imported file digests are in `source_manifest.json`; final distributed file hashes are in `RELEASE_MANIFEST.json`. The complete original handoff remains local. Missing historical tree predictions and historical pose/Judge/release runners have not been replaced by invented experiments.
