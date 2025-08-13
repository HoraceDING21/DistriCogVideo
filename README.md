# DistriCogVideo

Distributed text-to-video generation for CogVideoX models with patch-based parallelism.

DistriCogVideo mirrors the original CogVideoX pipeline API while adding efficient multi-GPU collaboration and memory-aware buffer preparation. It supports equal patch splitting, global K/V communication in attention, and automatic RoPE usage based on the loaded model.

## Requirements
- Python 3.10+
- PyTorch with CUDA (multi-GPU recommended for large models)
- Tested with `diffusers` CogVideoX pipelines

Install dependencies:

```bash
after cd CogVideo
pip install -r requirements.txt
```

## Quickstart

### Single GPU (not tested yet)
```bash
python run_distri_cogvideo.py \
  --model_path THUDM/CogVideoX-2b \
  --prompt "A cat playing with a ball" \
  --num_inference_steps 30 \
  --output_path output.mp4
```

### Multi-GPU (recommended)
```bash
torchrun --nproc_per_node=2 run_distri_cogvideo.py \
  --model_path THUDM/CogVideoX-5b \
  --prompt "A futuristic city at sunset" \
  --num_inference_steps 30 \
  --output_path output_5b.mp4
```

Notes:
- RoPE is automatically handled based on the model weights. For example, `CogVideoX-5B` uses RoPE; `CogVideoX-2B` does not.
- Only t2v is supported in this repo (no i2v/v2v here).

## CLI Arguments
These match the original CogVideoX demo where applicable, plus a minimal set for distributed operation.

Core options:
- `--model_path` (str): HF model id or local path, e.g. `THUDM/CogVideoX-2b`, `THUDM/CogVideoX-5b`
- `--prompt` (str): Text prompt
- `--negative_prompt` (str): Negative prompt (optional)
- `--height` (int): Output height, default 480
- `--width` (int): Output width, default 720
- `--num_frames` (int): Number of frames, default 49
- `--fps` (int): Frames per second for saved video, default 8
- `--num_inference_steps` (int): Denoising steps, default 50
- `--guidance_scale` (float): CFG scale, default 6.0
- `--max_sequence_length` (int): Text seq len, default 226
- `--scheduler` (str): `DPM` (default) or `DDIM`
- `--dtype` (str): `float16` or `bfloat16` (default)
- `--output_path` (str): Video save path
- `--verbose`: Verbose logging

Distributed options:
- `--mode` (str): `full_sync`, `optimized` (default), or `no_sync`
- `--warmup_steps` (int): Steps to force sync at start, default 4
- `--comm_checkpoint` (int): Queue length before async all_gather, default 60

## Programmatic Usage
```python
from distri_pipeline_cogvideox import DistriCogVideoXPipeline
from distri_cogvideo_utils import DistriCogVideoConfig
import torch

config = DistriCogVideoConfig(
    height=480,
    width=720,
    num_frames=49,
    mode="optimized",
    warmup_steps=4,
    comm_checkpoint=60,
    verbose=True,
)

pipe = DistriCogVideoXPipeline.from_pretrained(
    pretrained_model_name_or_path="THUDM/CogVideoX-5b",
    distri_config=config,
    torch_dtype=torch.bfloat16,
)

# Optional: run once to register buffers on multi-GPU
pipe.prepare()

video = pipe(
    prompt="A cat playing with a ball",
    num_inference_steps=30,
    output_type="pil",
).frames[0]
```

## Model Compatibility
- `THUDM/CogVideoX-2b`: no RoPE (uses 3D sin-cos)
- `THUDM/CogVideoX-5b`: RoPE
- `THUDM/CogVideoX1.5-5B`: RoPE

RoPE is not user-controlled here; the model config decides. Using RoPE with a non-RoPE model will lead to poor results, so we avoid manual toggles.

## Tips
- For 5B/1.5-5B models, use multi-GPU for speed and memory headroom
- Keep `height` and `width` multiples of 16
- If you see first-run comm buffer logs, they are expected during `prepare()` or the first `__call__`

## Troubleshooting
- NCCL warnings on single GPU: harmless; distributed comm will be skipped
- Connect timeouts when downloading: ensure network access to HuggingFace
- Out of memory: try `--num_inference_steps` lower, use more GPUs, or use `bfloat16`

## License
See `LICENSE` and the upstream model licenses in `MODEL_LICENSE`.

## Acknowledgements
- CogVideoX by THUDM / ZhipuAI
- Hugging Face Diffusers team
- Concepts adapted from FLUX distributed attention patterns
