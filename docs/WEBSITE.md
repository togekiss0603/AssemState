# Project website

Public URL: https://togekiss0603.github.io/AssemState/

The source lives in `website/` on `main`. GitHub Pages publishes the root of
the `gh-pages` branch, which contains only that directory. `.nojekyll` keeps
the site as plain static HTML. Asset URLs are relative so the project path works.

After reviewing and committing website changes on `main`, publish with:

```bash
python scripts/audit_paper_alignment.py
git push origin main
git subtree split --prefix website -b gh-pages
git push origin gh-pages
```

In the repository's Pages settings, the source is “Deploy from a branch”,
branch `gh-pages`, folder `/ (root)`. No custom Actions workflow or build
dependencies are needed. Check the Pages deployment before announcing an update.
