# Migration steps

Run these commands from your local machine after this branch is checked out.

## 1) Create the new public GitHub repository

```bash
gh repo create haibaraaaaai/anisotropy-rotation-analysis --public --description "Anisotropy orientation and rotation analysis assets extracted from pyqtrod" --confirm
```

## 2) Initialize local git inside the extracted directory

```bash
cd anisotropy-rotation-analysis
git init -b main
git add .
git commit -m "Initial import from pyqtrod: anisotropy rotation analysis assets"
git remote add origin git@github.com:haibaraaaaai/anisotropy-rotation-analysis.git
```

If you prefer HTTPS remote:

```bash
git remote add origin https://github.com/haibaraaaaai/anisotropy-rotation-analysis.git
```

## 3) Push to GitHub

```bash
git push -u origin main
```

## 4) Optional: tag pyqtrod as pre-thesis-reorg (without deleting files)

From the `pyqtrod` repository root:

```bash
git tag -a pre-thesis-reorg -m "Snapshot before thesis repository split"
git push origin pre-thesis-reorg
```
