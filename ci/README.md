# Offline CI template

`smoke.yml` is the tested offline workflow template. To activate GitHub Actions, place it at `.github/workflows/smoke.yml` using an account/token with workflow-write permission. The initial upload credential has repository access but no `workflow` OAuth scope, so this release keeps the template here rather than claiming an active CI run.

Run the same checks locally:

```bash
python -m pip install -r requirements-lock.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/reproduce_tables.py
```

The release-file check scans for accidental credentials and machine-specific paths; it does not require a generated inventory.
