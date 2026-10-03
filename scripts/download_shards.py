"""Download dataset shards (run on a machine with Hugging Face access). NOT tested in the build sandbox.
usage: python scripts/download_shards.py [n_shards=64]   # start with 4-8 to iterate fast"""
import sys
from huggingface_hub import list_repo_files, hf_hub_download
repo = "marin-dna/genomes-v5-genome_set-humans-intervals-v18_255_128"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".jsonl.zst"))[:n]
for f in files:
    print(hf_hub_download(repo, f, repo_type="dataset", local_dir="data/raw"))
