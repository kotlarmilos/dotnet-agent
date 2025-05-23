#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def iter_commit_rows(snapshot_dir: Path, diff_dir: Path, repo: str):
    # Generator yielding single-row dicts
    for snapshot_path in sorted(snapshot_dir.glob('pr-*.json')):
        pr = json.loads(snapshot_path.read_text(encoding='utf-8'))
        pr_number = pr.get('number')
        if pr_number is None:
            continue

        commits = pr.get('commits', {}).get('nodes', [])
        for node in commits:
            commit = node.get('commit', {})
            oid = commit.get('oid')
            if oid:
                diff_file = diff_dir / f"{oid}.diff"
                commit['diff'] = diff_file.read_text(encoding='utf-8').strip() if diff_file.exists() else ''

        events = []
        # collect events
        for c in pr.get('comments', {}).get('nodes', []):
            ts = c.get('createdAt')
            if ts: events.append(('comment', ts, c))
        for rt in pr.get('reviewThreads', {}).get('nodes', []):
            for r in rt.get('comments', {}).get('nodes', []):
                ts = r.get('createdAt')
                if ts: events.append(('review', ts, r))
        for node in commits:
            ts = node.get('commit', {}).get('committedDate')
            if ts: events.append(('commit', ts, node))
        events.sort(key=lambda e: e[1])

        history = []
        for kind, ts, data in events:
            history.append((kind, ts, data))
            if kind != 'commit':
                continue

            c = data['commit']
            oid = c.get('oid')
            diff_text = c.get('diff', '')
            msg = c.get('message', '')
            if not oid or not diff_text.strip():
                continue

            # build prompt
            prompt_parts = [
                f"Title: {pr.get('title', '')}",
                f"Body: {pr.get('body', '')}",
            ]
            labels = pr.get('labels') or []
            if labels:
                prompt_parts.append("Labels: " + ", ".join(labels))

            # events since last commit
            last_idx = next((i for i in range(len(history)-2, -1, -1) if history[i][0] == 'commit'), None)
            seg = history[last_idx+1:-1] if last_idx is not None else history[:-1]
            if last_idx is not None:
                prev = history[last_idx][2]['commit']
                prompt_parts.append(
                    f"Last commit: {prev.get('message')}\nDiff:\n{prev.get('diff', '')}"
                )

            for ekind, _, edata in seg:
                if ekind == 'comment':
                    body = edata.get('body', '').strip()
                    if body:
                        prompt_parts.append(f"Comment: {body}")
                elif ekind == 'review':
                    path = edata.get('path', '')
                    review_body = edata.get('body', '').strip()
                    hunk = (edata.get('diffHunk') or '').strip()
                    prompt_parts.append(
                        f"Review on {path}: {review_body}\nDiff:\n{hunk}"
                    )

            author = c.get('author', {}) or {}
            yield {
                'prompt': '\n'.join(prompt_parts),
                'completion': f"Diff:\n{diff_text}",
                'repo': repo,
                'pr_number': pr_number,
                'title': pr.get('title', ''),
                'body': pr.get('body', ''),
                'created_at': pr.get('createdAt', ''),
                'closed_at': pr.get('closedAt', ''),
                'merged_at': pr.get('mergedAt', ''),
                'author': author.get('login', ''),
                'state': pr.get('state', ''),
                'additions': pr.get('additions', 0),
                'deletions': pr.get('deletions', 0),
                'changed_files': pr.get('changedFiles', 0),
                'head_ref': pr.get('headRefName', ''),
                'labels': ", ".join(labels),
                'completion_commit': oid,
            }

def main():
    BASE_DIR = Path(__file__).resolve().parent
    snapshot_dir = BASE_DIR.parent / 'data' / 'raw-data' / 'prs'
    diff_dir = BASE_DIR.parent / 'data' / 'raw-data' / 'diffs'
    dataset_dir = BASE_DIR.parent / 'data' / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # define schema
    schema = pa.schema([
        ('prompt', pa.string()),
        ('completion', pa.string()),
        ('repo', pa.string()),
        ('pr_number', pa.int64()),
        ('title', pa.string()),
        ('body', pa.string()),
        ('created_at', pa.string()),
        ('closed_at', pa.string()),
        ('merged_at', pa.string()),
        ('author', pa.string()),
        ('state', pa.string()),
        ('additions', pa.int64()),
        ('deletions', pa.int64()),
        ('changed_files', pa.int64()),
        ('head_ref', pa.string()),
        ('labels', pa.string()),
        ('completion_commit', pa.string()),
    ])

    train_writer = pq.ParquetWriter(str(dataset_dir / 'train.parquet'), schema)
    test_writer  = pq.ParquetWriter(str(dataset_dir / 'test.parquet'), schema)

    for row in iter_commit_rows(snapshot_dir, diff_dir, 'dotnet/runtime'):
        table = pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema)
        # route by hash of commit ID
        if hash(row['completion_commit']) % 5 == 0:
            test_writer.write_table(table)
        else:
            train_writer.write_table(table)

    train_writer.close()
    test_writer.close()

    print(f"Wrote train.parquet and test.parquet to {dataset_dir}")

if __name__ == '__main__':
    main()