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
    settings = load_settings(BASE_DIR / 'settings.json')
    system_instruction = settings['system_instruction']
    model_name = settings.get('base_model')
    max_context = settings.get('max_context_size')

    # Define directories
    SNAPSHOT_DIR = BASE_DIR.parent / 'data' / 'raw-data' / 'prs'
    DIFF_DIR     = BASE_DIR.parent / 'data' / 'raw-data' / 'diffs'
    OUT_DIR      = BASE_DIR.parent / 'data' / 'dataset'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for snapshot_path in sorted(SNAPSHOT_DIR.glob('pr-*.json')):
        pr_data = json.loads(snapshot_path.read_text(encoding='utf-8'))
        pr_number = pr_data.get('prNumber')
        if pr_number is None:
            print(f"Skipping {snapshot_path.name}: missing prNumber")
            continue

        out_file = OUT_DIR / f"pr-{pr_number}.{output_format}"
        if out_file.exists():
            print(f"Skipping PR #{pr_number} (already exists)")
            continue

        # Extract PR metadata
        repo           = 'dotnet/runtime'
        title          = pr_data.get('title', '')
        body           = pr_data.get('body', '')
        labels         = pr_data.get('labels', [])
        comments       = pr_data.get('comments', [])
        review_comments= pr_data.get('reviewComments', [])
        commits_meta   = pr_data.get('commitsMeta', [])

        # Attach diffs
        for cm in commits_meta:
            oid = cm.get('commit', {}).get('oid')
            if oid:
                diff_file = DIFF_DIR / f"{oid}.diff"
                if diff_file.exists():
                    cm['commit']['diff'] = diff_file.read_text(encoding='utf-8').strip()

        # Build sorted events
        events = []
        for c in comments:
            ts = c.get('createdAt')
            if ts: events.append(('comment', ts, c))
        for rc in review_comments:
            ts = rc.get('createdAt')
            if ts: events.append(('review', ts, rc))
        for cm in commits_meta:
            ts = cm.get('commit', {}).get('committedDate')
            if ts: events.append(('commit', ts, cm))
        events.sort(key=lambda e: e[1])

        past = []
        lines = []
        csv_rows = []

        for kind, ts, data in events:
            if kind == 'commit':
                commit_info = data['commit']
                oid         = commit_info.get('oid')
                diff_text   = commit_info.get('diff', '')
                commit_date = commit_info.get('committedDate', '')
                commit_author = (commit_info.get('author') or {}).get('login', '')

                # Skip if commit diff is empty or not found
                if not diff_text.strip():
                    print(f"Skipping commit {oid}: diff is empty or not found.")
                    continue

                # Build user message (make prompt smaller)
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
                completion = f"Commit {oid}\nDiff:\n{diff_text}"

                # Tokenize prompt and completion separately
                prompt_tokens = tokenizer.encode(prompt, truncation=False)
                completion_tokens = tokenizer.encode(completion, truncation=False)
                # If completion alone is above max_context, skip
                if len(completion_tokens) > max_context:
                    print(f"Skipping commit {oid}: completion length = {len(completion_tokens)} tokens > {max_context}")
                    continue
                # If total is above max_context, truncate prompt from the start
                total_tokens = len(prompt_tokens) + len(completion_tokens)
                if total_tokens > max_context:
                    # Truncate prompt tokens from the start
                    num_prompt_tokens_allowed = max_context - len(completion_tokens)
                    if num_prompt_tokens_allowed > 0:
                        truncated_prompt_tokens = prompt_tokens[-num_prompt_tokens_allowed:]
                        truncated_prompt = tokenizer.decode(truncated_prompt_tokens, skip_special_tokens=True)
                        prompt = truncated_prompt
                    else:
                        print(f"Skipping commit {oid}: not enough room for prompt after reserving completion tokens.")
                        continue
                # Rebuild text with possibly truncated prompt
                text = (
                    "<|im_start|>system\n<|im_sep|>" + system_instruction + "\n<|im_end|>\n"
                    "<|im_start|>user\n<|im_sep|>"   + prompt + "\n<|im_end|>\n"
                    "<|im_start|>assistant\n<|im_sep|>" + completion + "\n<|im_end|>"
                )

                row = {
                    'prompt': prompt,
                    'completion': completion
                }
                if repo:
                    row['repo'] = repo
                if commit_author:
                    row['author'] = commit_author
                if commit_date:
                    row['date'] = commit_date
                if oid:
                    row['id'] = oid
                if labels:
                    row['labels'] = ','.join(labels)
                if output_format == 'jsonl':
                    lines.append(json.dumps(row, ensure_ascii=False))
                elif output_format == 'csv':
                    csv_rows.append(row)
            past.append((kind, ts, data))

        if output_format == 'jsonl' and lines:
            out_file.write_text("\n".join(lines) + "\n", encoding='utf-8')
            print(f"Generated {len(lines)} examples for PR #{pr_number} (jsonl)")
        elif output_format == 'csv' and csv_rows:
            # Dynamically determine fieldnames from the first row
            fieldnames = list(csv_rows[0].keys()) if csv_rows else ['prompt', 'completion']
            with out_file.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Generated {len(csv_rows)} examples for PR #{pr_number} (csv)")

if __name__ == "__main__":
    main()