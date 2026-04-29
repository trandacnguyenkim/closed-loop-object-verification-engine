
import os
import time

import torch
import traceback

# Keep HF cache on scratch by default on cluster. Can be overridden by env var.
os.environ.setdefault("HF_HOME", "/scratch/kt68/hf_cache")

_MIGC_SAFETENSOR = os.getenv("MIGC_SAFETENSOR", "")
_MIGC_YAML       = os.getenv("MIGC_YAML", "./pretrained_weights/v1-inference.yaml")
_MIGC_CKPT_PATH  = os.getenv("MIGC_CKPT_PATH", "./pretrained_weights/MIGC_SD14.ckpt")

_PIPE = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# CFG batch-size patch
# ---------------------------------------------------------------------------
# When guidance_scale > 1, diffusers stacks the conditional and unconditional
# UNet passes into a single forward call, doubling the batch dimension of
# hidden_states (and therefore query).  MIGCProcessor computes its cross-
# attention key tensor for only the conditional half, producing the error:
#
#   RuntimeError: Expected size for first two dimensions of batch2 tensor to
#   be: [32, 40] but got: [16, 40].
#
# The patch intercepts attn.get_attention_scores for the duration of each
# MIGCProcessor forward pass and tiles key (and optional mask) along the
# batch axis whenever query.shape[0] != key.shape[0].
# ---------------------------------------------------------------------------

def _patch_migc_for_cfg() -> None:
    """Monkey-patch MIGCProcessor to handle CFG batch doubling."""
    from migc.migc_pipeline import MIGCProcessor

    if getattr(MIGCProcessor, "_cfg_patched", False):
        return  # already applied

    _original_call = MIGCProcessor.__call__

    def _patched_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        # Wrap get_attention_scores to fix the batch-size mismatch on the fly.
        _orig_get_scores = attn.get_attention_scores

        def _safe_get_scores(query, key, mask=None):
            if key.shape[0] != query.shape[0]:
                # CFG doubled query; tile key (and mask) to match.
                factor = query.shape[0] // key.shape[0]
                key = key.repeat(factor, 1, 1)
                if mask is not None and mask.shape[0] != query.shape[0]:
                    repeat_dims = (factor,) + (1,) * (mask.dim() - 1)
                    mask = mask.repeat(*repeat_dims)
            return _orig_get_scores(query, key, mask)

        attn.get_attention_scores = _safe_get_scores
        try:
            result = _original_call(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            # Always restore, even if an exception occurs mid-call.
            attn.get_attention_scores = _orig_get_scores

        return result

    MIGCProcessor.__call__ = _patched_call
    MIGCProcessor._cfg_patched = True
    print("[ImageGenerator] MIGCProcessor CFG patch applied.")


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

_MIGC_SD_MODEL   = os.getenv("MIGC_SD_MODEL",   "CompVis/stable-diffusion-v1-4")
_MIGC_SAFETENSOR = os.getenv("MIGC_SAFETENSOR", "")
_MIGC_YAML       = os.getenv("MIGC_YAML",       "./pretrained_weights/v1-inference.yaml")
_MIGC_CKPT_PATH  = os.getenv("MIGC_CKPT_PATH",  "./pretrained_weights/MIGC_SD14.ckpt")
_MIGC_STEPS      = int(os.getenv("MIGC_STEPS",            "20"))
_INFERENCE_STEPS = int(os.getenv("IMAGE_INFERENCE_STEPS", "50"))


def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    try:
        from diffusers import EulerDiscreteScheduler
        from migc.migc_pipeline import AttentionStore, MIGCProcessor, StableDiffusionMIGCPipeline
        from migc.migc_utils import load_migc
    except ImportError as exc:
        raise RuntimeError("MIGC dependencies missing. Run `pip install -e .` in the MIGC repo.") from exc

    if not os.path.isfile(_MIGC_CKPT_PATH):
        raise FileNotFoundError(f"MIGC checkpoint not found at '{_MIGC_CKPT_PATH}'.")

    # Use cetusMix safetensor if available, otherwise fall back to HF base model
    if _MIGC_SAFETENSOR and os.path.isfile(_MIGC_SAFETENSOR):
        print(f"[ImageGenerator] Loading from safetensor: {_MIGC_SAFETENSOR}")
        from transformers import CLIPTextModel, CLIPTokenizer
        pipe = StableDiffusionMIGCPipeline.from_single_file(
            _MIGC_SAFETENSOR,
            original_config_file = _MIGC_YAML,
            load_safety_checker  = False,
        )
    else:
        print(f"[ImageGenerator] Loading base model: {_MIGC_SD_MODEL}")
        pipe = StableDiffusionMIGCPipeline.from_pretrained(_MIGC_SD_MODEL)

    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store, _MIGC_CKPT_PATH, attn_processor=MIGCProcessor)
    pipe = pipe.to(_DEVICE)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    _patch_migc_for_cfg()

    _PIPE = pipe
    return _PIPE


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "watermark, logo, signature, text, caption, subtitles, brand name, username, stamp, "
    "copyright, overlay, label, writing, letters, numbers, font, typeset, "
    "worst quality, low quality, bad anatomy, blurry, out of focus"
)


def generate_grounded_image(
    prompt: str,
    img_name: str,
    entity: list[str] = ["a cat", "a dog"],
    boxes: list[list[float]] = [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]],
    seed: int = 42,
    output_dir: str = "./dataset",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> str:
    """
    Generate a spatially-grounded image with MIGC and save it to disk.

    MIGC prompt format : [[global_prompt, instance_1, instance_2, ...]]
    MIGC boxes format  : [[[xmin, ymin, xmax, ymax], ...]]   (normalised 0-1)

    Args:
        prompt:          Global scene description.
        img_name:        Base filename (no extension) for the saved image/labels.
        entity:          Per-instance text labels aligned with `boxes`.
        boxes:           Normalised [xmin, ymin, xmax, ymax] per instance.
        seed:            RNG seed for the diffusion generator.
        output_dir:      Root folder; images/ and labels/ subdirs are created here.
        negative_prompt: Negative conditioning string.

    Returns:
        Absolute path to the saved image file.
    """
    pipe = _load_pipeline()

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    generator = torch.Generator(device=_DEVICE).manual_seed(seed)

    # MIGC prompt format: [[global, inst_0, inst_1, ...]]
    # MIGC boxes format:  [[[xmin, ymin, xmax, ymax], ...]]
    prompt_final = [[prompt] + list(entity)]
    bboxes       = [boxes]

    start_time = time.time()


    try:
        output = pipe(
            prompt              = prompt_final,
            bboxes              = bboxes,
            num_inference_steps = _INFERENCE_STEPS,
            guidance_scale      = 7.5,
            MIGCsteps           = _MIGC_STEPS,
            aug_phase_with_and  = False,
            negative_prompt     = negative_prompt,
            generator           = generator,
        )
    except Exception as e:
        traceback.print_exc()   
        raise


    elapsed = time.time() - start_time

    image = output.images[0]

    image_path = os.path.join(images_dir, f"{img_name}.jpg")
    image.save(image_path)
    image.show()
    image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
    image.show()

    # One label file per instance: "<class> xmin ymin xmax ymax"
    for i, (ent, box) in enumerate(zip(entity, boxes)):
        label_path = os.path.join(labels_dir, f"{img_name}_{i}.txt")
        with open(label_path, "w") as f:
            f.write(f"{ent} {box[0]} {box[1]} {box[2]} {box[3]}")

    print(f"[ImageGenerator] Generated '{img_name}' in {elapsed:.1f}s")
    return image_path