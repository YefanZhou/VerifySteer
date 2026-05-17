from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "YefanZhou98/VerifySteer"
REPO_TYPE = "dataset"

ROOT = Path(__file__).parent  # directory containing this script

DOWNLOADS = [
    "steering_vector/**",
    "verify_mlp_probe_weights/**",
    "precomputed_results/**",
]

snapshot_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    local_dir=str(ROOT),
    allow_patterns=DOWNLOADS,
)

print(f"Downloaded artifacts to: {ROOT}")






