"""Apply episode labels to update the HuggingFace dataset metadata.

Reads the labels from episode_labels.json (produced by label_episodes.py),
downloads the current metadata from HuggingFace, updates tasks.jsonl,
episodes.jsonl, and info.json, then uploads the changes.

Usage:
    # Dry run (preview changes without uploading):
    python examples/excavator/apply_labels.py --dry-run

    # Apply and upload:
    python examples/excavator/apply_labels.py
"""

import argparse
import json
import os
import tempfile

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "verm11/excavator_v3"
LABEL_FILE = os.path.join(os.path.dirname(__file__), "episode_labels.json")

PROMPTS = {
    "1": "Scoop packing peanuts from large pool and dump into pool on the left",
    "2": "Scoop packing peanuts from large pool and dump into pool on the right",
    "3": "Scoop packing peanuts from large pool and dump into the smallest pool",
    "4": "Scoop packing peanuts from large pool and dump into the medium sized pool",
    "5": "Scoop packing peanuts from large pool and dump into small pool",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("HF_TOKEN")

    with open(LABEL_FILE) as f:
        raw_labels = json.load(f)

    labels = {int(k): v for k, v in raw_labels.items() if v != "s"}
    print(f"Loaded {len(labels)} labels from {LABEL_FILE}")

    # Download current metadata
    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", token=token)
    episodes_path = hf_hub_download(REPO_ID, "meta/episodes.jsonl", repo_type="dataset", token=token)

    with open(info_path) as f:
        info = json.load(f)

    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    # Build unique task list from labels
    used_prompts = sorted(set(labels.values()))
    task_list = [PROMPTS[k] for k in used_prompts]

    # Include the default prompt only if there are unlabeled episodes
    default_prompt = "Scoop packing peanuts from large pool and dump into small pool"
    unlabeled_count = sum(1 for ep in episodes if ep["episode_index"] not in labels)
    if unlabeled_count > 0 and default_prompt not in task_list:
        task_list.append(default_prompt)
    task_list.sort()

    task_to_index = {t: i for i, t in enumerate(task_list)}
    prompt_key_to_task_index = {k: task_to_index[PROMPTS[k]] for k in used_prompts}

    print(f"\nTask list ({len(task_list)} tasks):")
    for idx, task in enumerate(task_list):
        count = sum(
            1 for ep in episodes
            if labels.get(ep["episode_index"], None) is not None
            and prompt_key_to_task_index.get(labels[ep["episode_index"]], -1) == idx
        )
        default_count = sum(
            1 for ep in episodes
            if ep["episode_index"] not in labels and task == default_prompt
        )
        total_count = count + default_count
        print(f"  [{idx}] ({total_count:3d} eps) {task}")

    # Update episodes
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep_idx in labels:
            prompt_key = labels[ep_idx]
            task_str = PROMPTS[prompt_key]
        else:
            task_str = default_prompt

        ep["task_index"] = task_to_index[task_str]
        ep["task"] = task_str

    # Update info.json
    info["total_tasks"] = len(task_list)

    if args.dry_run:
        print("\n[DRY RUN] Would upload the following changes:")
        print(f"  tasks.jsonl: {len(task_list)} tasks")
        print(f"  episodes.jsonl: {len(episodes)} episodes updated")
        print(f"  info.json: total_tasks = {len(task_list)}")
        print("\nSample episodes:")
        for ep in episodes[:5]:
            print(f"  ep {ep['episode_index']}: task_index={ep['task_index']} \"{ep['task']}\"")
        print("  ...")
        for ep in episodes[-3:]:
            print(f"  ep {ep['episode_index']}: task_index={ep['task_index']} \"{ep['task']}\"")
        return

    # Write to temp files and upload
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        tasks_path = os.path.join(tmp, "tasks.jsonl")
        with open(tasks_path, "w") as f:
            for idx, task in enumerate(task_list):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        eps_path = os.path.join(tmp, "episodes.jsonl")
        with open(eps_path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        info_out = os.path.join(tmp, "info.json")
        with open(info_out, "w") as f:
            json.dump(info, f, indent=2)

        print("\nUploading updated metadata...")
        api.upload_file(path_or_fileobj=tasks_path, path_in_repo="meta/tasks.jsonl",
                        repo_id=REPO_ID, repo_type="dataset")
        api.upload_file(path_or_fileobj=eps_path, path_in_repo="meta/episodes.jsonl",
                        repo_id=REPO_ID, repo_type="dataset")
        api.upload_file(path_or_fileobj=info_out, path_in_repo="meta/info.json",
                        repo_id=REPO_ID, repo_type="dataset")

    print("Done! Dataset metadata updated on HuggingFace.")


if __name__ == "__main__":
    main()
