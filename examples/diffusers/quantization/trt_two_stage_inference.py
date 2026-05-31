import itertools
import os
import tensorrt as trt
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from torchvision import transforms

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
RESOLUTION = 512
SEED = 0

BASE_NUM_STEPS = 1
DIFF_NUM_STEPS = 3
TOTAL_STEPS = BASE_NUM_STEPS + DIFF_NUM_STEPS

IMAGE_INDEX = 225  # Default dataset index from infer_two_stage_flow.py
SCALING_FACTOR = 1.0  # Flux2TinyAutoEncoder scaling factor

# Absolute/relative paths for models and data
TRT_ENCODER_PATH = "flux_vae_tiny_trt_v2/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt_v2/vae_decoder.plan"
TRT_BASE_UNET_PATH = "sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v4/unet.plan"
TRT_DIFF_UNET_PATH = "sid_two_stage_v1_5500/unet.plan"

DELTA_STD_PATH = r"C:\programming\auto_remaster\inference_optimization\models\sid_two_stage_v1_5500\delta_std.pt"
DATASET_CACHE_DIR = r"C:\programming\auto_remaster\dataset\nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora"

OUTPUT_IMAGE_PATH = "inference_result_trt_two_stage.png"
# IMAGE_PATH = "test_input.jpg"  
IMAGE_PATH = None  

# -----------------------------------------------------------------------------
# Structured noise helpers
# -----------------------------------------------------------------------------
def create_frequency_soft_cutoff_mask(
    height: int,
    width: int,
    cutoff_radius: float,
    transition_width: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")
    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    mask = torch.exp(
        -((frequency_radius - cutoff_radius) ** 2) / (2 * transition_width**2)
    )
    return torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)


def clip_frequency_magnitude(magnitudes, clip_percentile=0.95):
    flat = magnitudes.flatten()
    max_elements = 1_000_000
    if flat.numel() > max_elements:
        indices = torch.randperm(flat.numel(), device=flat.device)[:max_elements]
        flat = flat[indices]
    clip_threshold = torch.quantile(flat, clip_percentile)
    return torch.clamp(magnitudes, max=clip_threshold)


def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise: torch.Tensor = None,
    sampling_method: str = "fft",
) -> torch.Tensor:
    assert sampling_method in ["fft", "cdf", "two-gaussian"]
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()

    pad_h = int(height * (pad_factor - 1)) // 2 * 2
    pad_w = int(width * (pad_factor - 1)) // 2 * 2
    padded_images = torch.nn.functional.pad(
        image_batch, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), mode="reflect"
    )
    padded_height = height + pad_h
    padded_width = width + pad_w

    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height / 2, padded_width / 2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        freq_mask = torch.ones(padded_height, padded_width, device=device)

    fft_shifted = torch.fft.fftshift(
        torch.fft.fft2(padded_images, dim=(-2, -1)), dim=(-2, -1)
    )
    image_phases = clip_frequency_magnitude(torch.angle(fft_shifted))
    image_magnitudes = torch.abs(fft_shifted)

    if input_noise is not None:
        noise_batch = torch.nn.functional.pad(
            input_noise,
            (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
            mode="reflect",
        ).float()
    else:
        noise_batch = torch.randn_like(padded_images)

    if sampling_method == "fft":
        noise_fft_shifted = torch.fft.fftshift(
            torch.fft.fft2(noise_batch, dim=(-2, -1)), dim=(-2, -1)
        )
        noise_magnitudes = torch.abs(noise_fft_shifted)
        noise_phases = torch.angle(noise_fft_shifted)
    elif sampling_method == "cdf":
        N = padded_height * padded_width
        rayleigh_scale = (N / 2) ** 0.5
        uu = torch.rand(size=image_magnitudes.shape, device=device)
        noise_magnitudes = rayleigh_scale * torch.sqrt(-2.0 * torch.log(uu))
        if input_noise is not None:
            noise_fft_shifted = torch.fft.fftshift(
                torch.fft.fft2(noise_batch, dim=(-2, -1)), dim=(-2, -1)
            )
            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = (
                torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi
                - torch.pi
            )
    elif sampling_method == "two-gaussian":
        N = padded_height * padded_width
        rayleigh_scale = (N / 2) ** 0.5
        u1 = torch.randn_like(image_magnitudes)
        u2 = torch.randn_like(image_magnitudes)
        noise_magnitudes = rayleigh_scale * torch.sqrt(u1**2 + u2**2)
        if input_noise is not None:
            noise_fft_shifted = torch.fft.fftshift(
                torch.fft.fft2(noise_batch, dim=(-2, -1)), dim=(-2, -1)
            )
            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = (
                torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi
                - torch.pi
            )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    noise_magnitudes = clip_frequency_magnitude(noise_magnitudes) * noise_std

    fm = freq_mask.unsqueeze(0).unsqueeze(0)
    mixed_phases = fm * image_phases + (1 - fm) * noise_phases
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)

    structured_noise_padded = torch.real(
        torch.fft.ifft2(torch.fft.ifftshift(fft_combined, dim=(-2, -1)), dim=(-2, -1))
    )

    clamp_mask = (
        (structured_noise_padded < -5) | (structured_noise_padded > 5)
    ).float()
    structured_noise_padded = (
        structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    )

    return structured_noise_padded[
        :, :, pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width
    ].to(dtype)

# -----------------------------------------------------------------------------
# TensorRT Wrapper
# -----------------------------------------------------------------------------
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def _load_engine(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"TRT engine not found: {path}")
        with open(path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, feed_dict):
        outputs = {}
        input_tensors = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name not in feed_dict:
                    raise ValueError(f"Missing input '{name}', got: {list(feed_dict)}")
                t = feed_dict[name].half().contiguous()
                input_tensors.append(t)
                self.context.set_input_shape(name, t.shape)
                self.context.set_tensor_address(name, t.data_ptr())
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(shape, dtype=torch.float16, device="cuda")
                self.context.set_tensor_address(name, out.data_ptr())
                outputs[name] = out

        self.stream.wait_stream(torch.cuda.current_stream())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self._live_inputs = input_tensors
        torch.cuda.current_stream().wait_stream(self.stream)
        return outputs

# -----------------------------------------------------------------------------
# Euler Scheduler Helper
# -----------------------------------------------------------------------------
def get_euler_schedule(num_steps, device):
    sigmas_inner = torch.linspace(
        1.0, 1.0 / num_steps, num_steps, device=device, dtype=torch.float16
    )
    sigmas = torch.cat([sigmas_inner, sigmas_inner.new_zeros(1)])  # append 0
    dt = sigmas[1:] - sigmas[:-1]
    timesteps = sigmas_inner * 1000
    return dt, timesteps

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load input image
    target_display = None
    if IMAGE_PATH:
        print(f"Loading image from {IMAGE_PATH}...")
        orig_source_pil = Image.open(IMAGE_PATH).convert("RGB")
    else:
        print(f"Loading dataset item {IMAGE_INDEX}...")
        dataset = load_dataset(
            "dim/render_nfs_4screens_5_sdxl_1_wan_mix", 
            split="train",
            streaming=True
        )
        item = next(itertools.islice(dataset, IMAGE_INDEX, None))
        orig_source_pil = item["input_image"].convert("RGB")
        target_pil = item["edited_image"].convert("RGB")
        
        target_preprocess = transforms.Compose([
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(RESOLUTION),
        ])
        target_display = target_preprocess(target_pil)

    # 2. Load engines
    print("Loading TensorRT engines...")
    try:
        enc_engine = TRTEngine(TRT_ENCODER_PATH)
        dec_engine = TRTEngine(TRT_DECODER_PATH)
        unet_base_engine = TRTEngine(TRT_BASE_UNET_PATH)
        unet_diff_engine = TRTEngine(TRT_DIFF_UNET_PATH)
    except Exception as e:
        print(f"Error loading engines: {e}")
        exit(1)
    print("Engines loaded.")

    # 3. Load delta_std.pt
    print(f"Loading delta_std from {DELTA_STD_PATH}...")
    if not os.path.exists(DELTA_STD_PATH):
        print(f"Error: delta_std.pt not found at absolute path: {DELTA_STD_PATH}")
        exit(1)
    delta_std = torch.load(DELTA_STD_PATH, map_location="cpu").to(
        device=DEVICE, dtype=torch.float16
    )  # shape expected: [1, 32, 1, 1]
    print(f"Loaded delta_std, shape={delta_std.shape}, mean={delta_std.mean().item():.4f}")

    # 4. Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    c_t = preprocess(orig_source_pil).unsqueeze(0).to(DEVICE).half()
    source_display = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
    ])(orig_source_pil)

    # Encode source image to latent space
    z_source = list(enc_engine.infer({"image": c_t}).values())[0] * SCALING_FACTOR
    print(f"z_source shape: {z_source.shape}")

    # 5. Baseline Path: Base Model ONLY with total step budget (fair comparison)
    print(f"\n=== Running Baseline: Base Model ONLY ({TOTAL_STEPS} steps) ===")
    dt_baseline, timesteps_baseline = get_euler_schedule(TOTAL_STEPS, DEVICE)

    # Build structured noise for baseline
    input_noise_baseline = torch.randn(z_source.shape, device=DEVICE, dtype=torch.float32)
    sample_baseline = generate_structured_noise_batch_vectorized(
        z_source.float(),
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=100.0,
        input_noise=input_noise_baseline,
        sampling_method="fft",
    ).to(dtype=z_source.dtype, device=DEVICE)

    # Diffusion Loop for Baseline
    for i in range(TOTAL_STEPS):
        cat_input = torch.cat([sample_baseline, z_source], dim=1)
        t_tensor = timesteps_baseline[i].expand(cat_input.shape[0])
        pred = list(
            unet_base_engine.infer({"sample": cat_input, "timestep": t_tensor}).values()
        )[0]
        sample_baseline = sample_baseline + dt_baseline[i] * pred

    z_baseline = sample_baseline
    print(f"z_baseline shape: {z_baseline.shape}")

    # 6. Stage 1: Base Model (UNet v5), BASE_NUM_STEPS steps
    print(f"\n=== Running Stage 1: Base Model ({BASE_NUM_STEPS} steps) ===")
    dt_base, timesteps_base = get_euler_schedule(BASE_NUM_STEPS, DEVICE)

    # Build structured noise for Stage 1
    input_noise = torch.randn(z_source.shape, device=DEVICE, dtype=torch.float32)
    sample = generate_structured_noise_batch_vectorized(
        z_source.float(),
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=100.0,
        input_noise=input_noise,
        sampling_method="fft",
    ).to(dtype=z_source.dtype, device=DEVICE)

    # Diffusion Loop for Stage 1
    for i in range(BASE_NUM_STEPS):
        cat_input = torch.cat([sample, z_source], dim=1)
        t_tensor = timesteps_base[i].expand(cat_input.shape[0])
        pred = list(
            unet_base_engine.infer({"sample": cat_input, "timestep": t_tensor}).values()
        )[0]
        sample = sample + dt_base[i] * pred

    z_base = sample
    print(f"z_base shape: {z_base.shape}")

    # 7. Stage 2: Residual Model (UNet diff), DIFF_NUM_STEPS steps
    print(f"\n=== Running Stage 2: Residual Model ({DIFF_NUM_STEPS} steps) ===")
    dt_diff, timesteps_diff = get_euler_schedule(DIFF_NUM_STEPS, DEVICE)

    # Start from pure Gaussian noise
    sample_delta = torch.randn_like(z_base)

    # Diffusion Loop for Stage 2
    for i in range(DIFF_NUM_STEPS):
        # Conditioning: cat([noisy_delta, z_base])
        cat_input_diff = torch.cat([sample_delta, z_base], dim=1)
        t_tensor_diff = timesteps_diff[i].expand(cat_input_diff.shape[0])
        pred = list(
            unet_diff_engine.infer({"sample": cat_input_diff, "timestep": t_tensor_diff}).values()
        )[0]
        sample_delta = sample_delta + dt_diff[i] * pred

    # Recombine: z_final = z_base + delta * delta_std
    delta_pred = sample_delta * delta_std
    z_final = z_base + delta_pred
    print(f"delta_pred std: {delta_pred.std().item():.4f}")
    print(f"z_final shape: {z_final.shape}")

    # 8. Decode latents using TRT VAE Decoder
    print("\nDecoding latents to PIL images...")
    def decode_latent(latent):
        lat = latent / SCALING_FACTOR
        img_out = list(dec_engine.infer({"latent": lat}).values())[0]
        img_clamp = img_out.clamp(-1, 1)[0]
        return transforms.ToPILImage()(img_clamp.cpu().float() * 0.5 + 0.5)

    baseline_pil = decode_latent(z_baseline)
    base_pil = decode_latent(z_base)
    final_pil = decode_latent(z_final)

    # 9. Build and save comparison grid
    panels = [
        np.array(source_display),
        np.array(baseline_pil),
        np.array(base_pil),
        np.array(final_pil),
    ]
    if target_display is not None:
        panels.append(np.array(target_display))

    grid = Image.fromarray(np.hstack(panels))
    grid.save(OUTPUT_IMAGE_PATH)
    
    print(f"\nSaved comparison grid to {OUTPUT_IMAGE_PATH}")
    print(
        f"Grid layout: Source | BaseOnly ({TOTAL_STEPS} steps) | Base ({BASE_NUM_STEPS} steps) "
        f"| Base+Delta ({BASE_NUM_STEPS}+{DIFF_NUM_STEPS} steps)"
        + (" | Target" if target_display is not None else "")
    )
