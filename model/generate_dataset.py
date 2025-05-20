#!/usr/bin/env python3
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer
import argparse
import csv

def load_settings(path: Path):
    if not path.exists():
        print(f"Settings file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for fine-tuning.")
    parser.add_argument('--output', choices=['jsonl', 'csv'], default='jsonl', help='Output format: jsonl or csv (default: jsonl)')
    args = parser.parse_args()
    output_format = args.output

    # Load settings
    BASE_DIR = Path(__file__).resolve().parent

    # Define directories
    SNAPSHOT_DIR = BASE_DIR.parent / 'data' / 'raw-data' / 'prs'
    DIFF_DIR     = BASE_DIR.parent / 'data' / 'raw-data' / 'diffs'
    OUT_DIR      = BASE_DIR.parent / 'data' / 'dataset'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for snapshot_path in sorted(SNAPSHOT_DIR.glob('pr-*.json')):
        pr_data = json.loads(snapshot_path.read_text(encoding='utf-8'))
        pr_number = pr_data.get('number')
        if pr_number is None:
            print(f"Skipping {snapshot_path.name}: missing number")
            continue

        # Extract PR metadata
        repo           = 'dotnet/runtime'
        title          = pr_data.get('title', '')
        body           = pr_data.get('body', '')
        created_at     = pr_data.get('createdAt', '')
        closed_at      = pr_data.get('closedAt', '')
        merged_at      = pr_data.get('mergedAt', '')
        author         = pr_data.get('author', {}).get('login', '')
        state          = pr_data.get('state', '')
        additions      = pr_data.get('additions', 0)
        deletions      = pr_data.get('deletions', 0)
        changed_files  = pr_data.get('changedFiles', 0)
        head_ref       = pr_data.get('headRefName', '')
        labels         = pr_data.get('labels', [])
        comments       = pr_data.get('comments', [])
        review_threads = pr_data.get('reviewThreads', [])
        commits        = pr_data.get('commits', [])

        # Attach diffs
        for cm in commits.get('nodes', []):
            oid = cm.get('commit', {}).get('oid')
            if oid:
                diff_file = DIFF_DIR / f"{oid}.diff"
                if diff_file.exists():
                    cm['commit']['diff'] = diff_file.read_text(encoding='utf-8').strip()

        # Build sorted events
        events = []
        for c in comments.get('nodes', []):
            ts = c.get('createdAt')
            if ts: events.append(('comment', ts, c))
        for rc in review_threads.get('nodes', []):
            for r in rc['comments'].get('nodes', []):
                ts = r.get('createdAt')
                if ts: events.append(('review', ts, r))
        for cm in commits.get('nodes', []):
            ts = cm.get('commit', {}).get('committedDate')
            if ts: events.append(('commit', ts, cm))
        events.sort(key=lambda e: e[1])

        past = []
        for kind, ts, data in events:
            if kind == 'commit':
                commit_info = data['commit']
                oid         = commit_info.get('oid')
                diff_text   = commit_info.get('diff', '')
                commit_date = commit_info.get('committedDate', '')
                commit_message = commit_info.get('message', '')

                # Skip if commit file already exists
                commit_file = OUT_DIR / f"commit-{oid}.jsonl"
                if commit_file.exists():
                    print(f"Skipping commit {oid}: file already exists.")
                    continue

                # Skip if commit diff is empty or not found
                if not diff_text.strip():
                    print(f"Skipping commit {oid}: diff is empty or not found.")
                    continue

                # Build user message
                user_parts = [
                    f"PR #{pr_number}: {title}",
                    f"Body: {body}",
                ]
                if labels:
                    user_parts.append("Labels: " + ", ".join(labels))
                # Find the last commit before the current one
                last_commit_idx = None
                for idx in range(len(past)-1, -1, -1):
                    if past[idx][0] == 'commit':
                        last_commit_idx = idx
                        break
                    
                # Gather events between last commit and this commit
                events_between = past[last_commit_idx+1:] if last_commit_idx is not None else past
                # Add last commit info if it exists before current commit
                if last_commit_idx is not None:
                    prev_commit_event = past[last_commit_idx]
                    prev_commit_info = (prev_commit_event[2] or {}).get('commit', {})
                    prev_oid = prev_commit_info.get('oid', '')
                    prev_message = prev_commit_info.get('message', '')
                    prev_diff = prev_commit_info.get('diff', '')
                    prev_date = prev_commit_info.get('committedDate', '')
                    if prev_date and commit_date and prev_date < commit_date:
                        user_parts.append(
                            f"Last commit {prev_oid} – {prev_message}\nDiff:\n{prev_diff}"
                        )
                # Add all comments and reviews between last commit and this commit
                for pkind, _, pdat in events_between:
                    if pkind == 'comment':
                        comment_body = pdat.get('body', '') if pdat else ''
                        if comment_body.strip():
                            user_parts.append(f"Comment: {comment_body}")
                    elif pkind == 'review':
                        path   = (pdat or {}).get('path', '')
                        hunk   = ((pdat or {}).get('diffHunk') or '').strip()
                        review_body = (pdat or {}).get('body', '')
                        if review_body.strip() or hunk:
                            user_parts.append(
                                f"Review on {path}: {review_body}\nDiff:\n{hunk}"
                            )
                prompt = "\n".join(user_parts)
                completion = f"Commit {oid} - {commit_message}\nDiff:\n{diff_text}"

                row = {
                    'prompt': prompt,
                    'completion': completion
                }
                if repo:
                    row['repo'] = repo
                if pr_number:
                    row['pr_number'] = pr_number
                if title:
                    row['title'] = title
                if body:
                    row['body'] = body
                if created_at:
                    row['created_at'] = created_at
                if closed_at:
                    row['closed_at'] = closed_at
                if merged_at:
                    row['merged_at'] = merged_at
                if author:
                    row['author'] = author
                if state:
                    row['state'] = state
                if additions:
                    row['additions'] = additions
                if deletions:
                    row['deletions'] = deletions
                if changed_files:
                    row['changed_files'] = changed_files
                if head_ref:
                    row['head_ref'] = head_ref
                if labels:
                    row['labels'] = ", ".join(labels)
                row['completion_commit'] = oid
                # Write out per-commit file
                commit_out_file = OUT_DIR / f"commit-{oid}.{output_format}"
                if output_format == 'jsonl':
                    commit_out_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding='utf-8')
                elif output_format == 'csv':
                    fieldnames = list(row.keys())
                    with commit_out_file.open('w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow(row)
            past.append((kind, ts, data))

if __name__ == "__main__":
    main()