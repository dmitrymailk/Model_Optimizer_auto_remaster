import itertools
import os

import tensorrt as trt
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms

# -----------------------------------------------------------------------------
# Structured noise helpers (mirrors flow_matching_inference_win.py)
# -----------------------------------------------------------------------------
# All engines (VAE enc/dec, UNet) are built with FP16 — no runtime dtype detection needed.


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
RESOLUTION = 512
NUM_STEPS = 4
IMAGE_INDEX = 170
SCALING_FACTOR = 1.0  # Flux2TinyAutoEncoder.config.scaling_factor

TRT_ENCODER_PATH = "flux_vae_tiny_trt_v2/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt_v2/vae_decoder.plan"
TRT_UNET_PATH = "sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3/unet.plan"

OUTPUT_IMAGE_PATH = "inference_result_trt_full.png"
IMAGE_PATH = "test_input.jpg"  # set to None to load from dataset instead


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

        # TRT ждёт PyTorch-операций (Euler-шаг предыдущей итерации)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        # Сохраняем ссылки — к следующему вызову infer предыдущий TRT-вызов
        # уже завершён: current.wait_stream(self.stream) → Euler → wait_stream(current)
        self._live_inputs = input_tensors
        # PyTorch-поток ждёт TRT перед чтением pred (Euler-шаг на current_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return outputs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load input image
    if IMAGE_PATH:
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

    # 2. Load engines
    print("Loading TensorRT engines...")
    try:
        enc_engine = TRTEngine(TRT_ENCODER_PATH)
        dec_engine = TRTEngine(TRT_DECODER_PATH)
        unet_engine = TRTEngine(TRT_UNET_PATH)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print("Engines loaded.")

    # 3. Preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    c_t = preprocess(orig_source_pil).unsqueeze(0).to(DEVICE).half()

    # 4. Pipeline
    print(f"Running TRT pipeline ({NUM_STEPS} steps)...")
    # Euler flow matching: x_next = x + (sigma_next - sigma_curr) * pred
    # Воспроизводим FlowMatchEulerDiscreteScheduler точно:
    #   sigmas = linspace(1.0, 1/N, N) затем append 0  → [1.0, 0.75, 0.5, 0.25, 0.0]
    #   timesteps = sigmas * 1000                       → [1000, 750, 500, 250]
    sigmas_inner = torch.linspace(
        1.0, 1.0 / NUM_STEPS, NUM_STEPS, device=DEVICE, dtype=torch.float16
    )
    sigmas = torch.cat([sigmas_inner, sigmas_inner.new_zeros(1)])  # append 0
    dt = sigmas[1:] - sigmas[:-1]  # [-0.25, -0.25, -0.25, -0.25]
    timesteps = sigmas_inner * 1000  # [1000, 750, 500, 250]

    # Encode
    z_source = list(enc_engine.infer({"image": c_t}).values())[0] * SCALING_FACTOR

    # Structured noise init
    input_noise = torch.randn(z_source.shape, device=DEVICE, dtype=torch.float32)
    sample = generate_structured_noise_batch_vectorized(
        z_source.float(),
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=100.0,
        input_noise=input_noise,
        sampling_method="fft",
    ).to(dtype=z_source.dtype, device=DEVICE)

    # Diffusion loop — чистый Euler без scheduler
    for i in range(NUM_STEPS):
        cat_input = torch.cat([sample, z_source], dim=1)
        t_tensor = timesteps[i].expand(cat_input.shape[0])
        pred = list(
            unet_engine.infer({"sample": cat_input, "timestep": t_tensor}).values()
        )[0]
        sample = sample + dt[i] * pred

    # Decode
    latents = sample / SCALING_FACTOR
    output_image = list(dec_engine.infer({"latent": latents}).values())[0].clamp(-1, 1)

    # Save
    image_tensor = output_image[0] if output_image.dim() == 4 else output_image
    output_pil = transforms.ToPILImage()(image_tensor.cpu().float() * 0.5 + 0.5)
    output_pil.save(OUTPUT_IMAGE_PATH)
    print(f"Saved → {OUTPUT_IMAGE_PATH}")
