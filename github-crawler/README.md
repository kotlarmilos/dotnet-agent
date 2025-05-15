# GitHub Crawler

A simple CLI tool to retrieve PR metadata, comments, reviews, and commit diffs from public GitHub repo.

## Quick start

1. **Install dependencies**:
```bash
npm install
```

2. **Set your GitHub token**:
```
export GITHUB_TOKEN=YOUR_TOKEN
```

3. **Run**:
```
node main.js --owner dotnet --repo runtime --outDir ../data/raw-data --concurrency 5 --maxPrs 10 --skipDiff
```

## Expected output
After running, you'll find:
```
output/
├── snapshots/
│   ├── pr-1.json
│   ├── pr-2.json
│   └── ...
└── diffs/
    ├── <sha1>.diff
    ├── <sha2>.diff
    └── ...
```