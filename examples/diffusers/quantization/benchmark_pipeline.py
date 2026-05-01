import itertools
import os
import time

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder

# -----------------------------------------------------------------------------
# Structured noise helpers (mirrors trt_full_inference.py)
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
    return torch.clamp(magnitudes, max=torch.quantile(magnitudes, clip_percentile))


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
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16
RESOLUTION = 512
NUM_STEPS = 4
IMAGE_INDEX = 170
SCALING_FACTOR = 1.0  # Flux2TinyAutoEncoder.config.scaling_factor

# Set to a local path to skip dataset download, or None to load from dataset
IMAGE_PATH = "test_input.jpg"

# Torch Paths
CHECKPOINT_PATH = r"C:\programming\auto_remaster\inference_optimization\models\sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3\student"

# TRT Paths
TRT_ENCODER_PATH = "flux_vae_tiny_trt_v2/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt_v2/vae_decoder.plan"
TRT_UNET_PATH = "sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3/unet.plan"

# Benchmark Config
WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 200


# -----------------------------------------------------------------------------
# TensorRT Engine (mirrors trt_full_inference.py — single context, stream sync)
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
# Pure Euler helpers (no scheduler — mirrors trt_full_inference.py)
# -----------------------------------------------------------------------------
def make_euler_sigmas(num_steps: int, device):
    """Returns (sigmas, dt, timesteps) — pure linspace Euler, no scheduler."""
    sigmas_inner = torch.linspace(
        1.0, 1.0 / num_steps, num_steps, device=device, dtype=torch.float16
    )
    sigmas = torch.cat([sigmas_inner, sigmas_inner.new_zeros(1)])  # append 0
    dt = sigmas[1:] - sigmas[:-1]          # e.g. [-0.25, -0.25, ...]
    timesteps = sigmas_inner * 1000        # e.g. [1000, 750, 500, 250]
    return sigmas_inner, dt, timesteps


def make_structured_noise(z_source):
    """Generate structured noise matched to z_source shape."""
    input_noise = torch.randn(z_source.shape, device=z_source.device, dtype=torch.float32)
    return generate_structured_noise_batch_vectorized(
        z_source.float(),
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=100.0,
        input_noise=input_noise,
        sampling_method="fft",
    ).to(dtype=z_source.dtype, device=z_source.device)


# -----------------------------------------------------------------------------
# Pipeline functions
# -----------------------------------------------------------------------------
def run_torch_pipeline(vae, unet, input_tensor, num_steps):
    with torch.no_grad():
        # Encode: VAE outputs 128ch latent; pixel_shuffle(2) → 32ch (UNet expects 32ch per branch)
        z_source = torch.nn.functional.pixel_shuffle(
            vae.encode(input_tensor).latent, 2
        ) * SCALING_FACTOR

        # Structured noise init
        sample = make_structured_noise(z_source)

        # Pure Euler loop
        _, dt, timesteps = make_euler_sigmas(num_steps, DEVICE)
        for i in range(num_steps):
            cat_input = torch.cat([sample, z_source], dim=1)  # 64ch total
            t_batch = timesteps[i].expand(cat_input.shape[0])
            pred = unet(cat_input, t_batch, return_dict=False)[0]
            sample = sample + dt[i] * pred

        # Decode: pixel_unshuffle(2) → 128ch before passing to VAE decoder
        latent = torch.nn.functional.pixel_unshuffle(sample / SCALING_FACTOR, 2)
        output = vae.decode(latent).sample.clamp(-1, 1)
        return output


def run_trt_pipeline(enc_engine, dec_engine, unet_engine, input_tensor, num_steps):
    # Encode
    z_source = list(enc_engine.infer({"image": input_tensor}).values())[0] * SCALING_FACTOR

    # Structured noise init
    sample = make_structured_noise(z_source)

    # Pure Euler loop
    _, dt, timesteps = make_euler_sigmas(num_steps, DEVICE)
    for i in range(num_steps):
        cat_input = torch.cat([sample, z_source], dim=1)
        t_tensor = timesteps[i].expand(cat_input.shape[0])
        pred = list(unet_engine.infer({"sample": cat_input, "timestep": t_tensor}).values())[0]
        sample = sample + dt[i] * pred

    # Decode
    latents = sample / SCALING_FACTOR
    output = list(dec_engine.infer({"latent": latents}).values())[0].clamp(-1, 1)
    return output


# -----------------------------------------------------------------------------
# Benchmark harness
# -----------------------------------------------------------------------------
def benchmark(name, func, *args):
    print(f"\nBenchmarking {name}...")

    # Warmup
    print(f"  Warmup ({WARMUP_ROUNDS} rounds)...")
    for _ in range(WARMUP_ROUNDS):
        func(*args)
    torch.cuda.synchronize()

    # Measure
    print(f"  Running {BENCHMARK_ROUNDS} rounds...")
    latencies = []
    last_output = None

    for _ in range(BENCHMARK_ROUNDS):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        last_output = func(*args)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(f"  {name}: Avg = {avg_latency:.2f} ms ± {std_latency:.2f} ms")
    return avg_latency, last_output


def save_image(tensor, path):
    """Save (1,3,H,W) or (3,H,W) tensor in [-1,1] to PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.cpu().float() * 0.5 + 0.5
    image = image.clamp(0, 1)
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(path)
    print(f"Saved → {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # 1. Prepare input image
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        print(f"Loading image from {IMAGE_PATH}...")
        orig_source_pil = Image.open(IMAGE_PATH).convert("RGB")
    else:
        print(f"Loading dataset item {IMAGE_INDEX}...")
        dataset = load_dataset(
            "dim/render_nfs_4screens_5_sdxl_1_wan_mix", split="train", streaming=True
        )
        orig_source_pil = next(itertools.islice(dataset, IMAGE_INDEX, None))[
            "input_image"
        ].convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = preprocess(orig_source_pil).unsqueeze(0).to(DEVICE).half()

    # 2. Init Torch Models
    print("Loading Torch models...")
    vae_torch = Flux2TinyAutoEncoder.from_pretrained(
        "dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora_v2",
        torch_dtype=WEIGHT_DTYPE,
    ).to(DEVICE).eval()

    unet_torch = UNet2DModel.from_pretrained(
        CHECKPOINT_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        use_safetensors=True,
    ).to(DEVICE).eval()

    # 3. Init TRT Engines
    print("Loading TensorRT engines...")
    enc_engine = TRTEngine(TRT_ENCODER_PATH)
    dec_engine = TRTEngine(TRT_DECODER_PATH)
    unet_engine = TRTEngine(TRT_UNET_PATH)

    # 4. Run Benchmarks
    torch_latency, torch_output = benchmark(
        "PyTorch Pipeline", run_torch_pipeline,
        vae_torch, unet_torch, input_tensor, NUM_STEPS,
    )
    save_image(torch_output, "benchmark_output_torch.png")

    trt_latency, trt_output = benchmark(
        "TensorRT Pipeline", run_trt_pipeline,
        enc_engine, dec_engine, unet_engine, input_tensor, NUM_STEPS,
    )
    save_image(trt_output, "benchmark_output_trt.png")

    # 5. Report
    torch_fps = 1000.0 / torch_latency
    trt_fps = 1000.0 / trt_latency
    print("\n---------------------------------------------------------")
    print("Final Results (End-to-End Latency)")
    print("---------------------------------------------------------")
    print(f"PyTorch:  {torch_latency:.2f} ms  ({torch_fps:.1f} FPS)")
    print(f"TensorRT: {trt_latency:.2f} ms  ({trt_fps:.1f} FPS)")
    if trt_latency > 0:
        print(f"Speedup:  {torch_latency / trt_latency:.2f}x")
    print("---------------------------------------------------------")
