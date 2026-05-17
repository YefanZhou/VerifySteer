from huggingface_hub import hf_hub_download
import shutil, tempfile
from pathlib import Path

REPO_ID = "YefanZhou98/vllm-wheel-steer"
VLLM_DIR = Path(__file__).parent / "EasySteer" / "vllm-steer" / "vllm"

SO_FILES = [
    "vllm_so/_C.abi3.so",
    "vllm_so/cumem_allocator.abi3.so",
    "vllm_so/_flashmla_C.abi3.so",
    "vllm_so/_flashmla_extension_C.abi3.so",
    "vllm_so/_moe_C.abi3.so",
    "vllm_so/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm_so/vllm_flash_attn/_vllm_fa3_C.abi3.so",
]

with tempfile.TemporaryDirectory() as tmp:
    for hf_path in SO_FILES:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=hf_path,
            local_dir=tmp,
        )
        rel = Path(hf_path).relative_to("vllm_so")
        dst = VLLM_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
        print(f"Installed: {rel}")

print("Done.")
