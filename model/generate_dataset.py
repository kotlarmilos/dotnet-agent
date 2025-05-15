#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def load_settings(path: Path):
    if not path.exists():
        print(f"Settings file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    # Load settings
    BASE_DIR = Path(__file__).resolve().parent
    settings = load_settings(BASE_DIR / 'settings.json')
    system_instruction = settings['system_instruction']
    model_name = settings.get('model_name')
    max_context = settings.get('max_context_size')

    # Define directories
    SNAPSHOT_DIR = BASE_DIR.parent / 'data' / 'raw-data' / 'prs'
    DIFF_DIR     = BASE_DIR.parent / 'data' / 'raw-data' / 'diffs'
    OUT_DIR      = BASE_DIR.parent / 'data' / 'dataset'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for snapshot_path in sorted(SNAPSHOT_DIR.glob('pr-*.json')):
        pr_data = json.loads(snapshot_path.read_text(encoding='utf-8'))
        pr_number = pr_data.get('prNumber')
        if pr_number is None:
            print(f"Skipping {snapshot_path.name}: missing prNumber")
            continue

        out_file = OUT_DIR / f"pr-{pr_number}.jsonl"
        if out_file.exists():
            print(f"Skipping PR #{pr_number} (already exists)")
            continue

        # Extract PR metadata
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

        for kind, ts, data in events:
            if kind == 'commit':
                commit_info = data['commit']
                oid         = commit_info.get('oid')
                diff_text   = commit_info.get('diff', '')

                # Build user message
                user_parts = [
                    f"PR #{pr_number} (model={model_name}): {title}",
                    f"Description: {body}"
                ]
                if labels:
                    user_parts.append("Labels: " + ", ".join(labels))

                # Add past events into context
                for pkind, _, pdat in past:
                    if pkind == 'comment':
                        author = pdat.get('author',{}).get('login','user')
                        user_parts.append(f"Comment by {author}: {pdat.get('body','')}")
                    elif pkind == 'review':
                        author = pdat.get('author',{}).get('login','reviewer')
                        path   = pdat.get('path','')
                        hunk   = pdat.get('diffHunk','').strip()
                        user_parts.append(
                            f"Review by {author} on {path}: {pdat.get('body','')}\nDiff hunk:\n{hunk}"
                        )
                    elif pkind == 'commit':
                        prev = pdat['commit']
                        user_parts.append(
                            f"Previous commit {prev.get('oid','')} – {prev.get('message','')}\n"
                            f"Diff:\n{prev.get('diff','')}"
                        )

                assistant_msg = f"Commit {oid}\nDiff:\n{diff_text}"
                text = (
                    "<|im_start|>system\n" + system_instruction + "\n<|im_end|>\n"
                    "<|im_start|>user\n"   + "\n".join(user_parts) + "\n<|im_end|>\n"
                    "<|im_start|>assistant\n" + assistant_msg + "\n<|im_end|>"
                )

                # Enforce max context size (in characters)
                if len(text) > max_context:
                    print(f"Skipping example for commit {oid}: context size {len(text)} > {max_context}")
                else:
                    lines.append(json.dumps({'text': text}, ensure_ascii=False))

            past.append((kind, ts, data))

        if lines:
            out_file.write_text("\n".join(lines) + "\n", encoding='utf-8')
            print(f"Generated {len(lines)} examples for PR #{pr_number}")

if __name__ == "__main__":
    main()