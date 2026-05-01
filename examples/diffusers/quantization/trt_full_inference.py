import os
import torch
import numpy as np
import tensorrt as trt
import itertools
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import FlowMatchEulerDiscreteScheduler


def create_frequency_soft_cutoff_mask(
    height: int,
    width: int,
    cutoff_radius: float,
    transition_width: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a smooth frequency cutoff mask for low-pass filtering.

    Args:
        height: Image height
        width: Image width
        cutoff_radius: Frequency cutoff radius (0 = no structure, max_radius = full structure)
        transition_width: Width of smooth transition (smaller = sharper cutoff)
        device: Device to create tensor on

    Returns:
        torch.Tensor: Frequency mask of shape (height, width)
    """
    if device is None:
        device = torch.device("cpu")

    # Create frequency coordinates
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")

    # Calculate distance from center
    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)

    # Create smooth transition mask
    mask = torch.exp(
        -((frequency_radius - cutoff_radius) ** 2) / (2 * transition_width**2)
    )
    mask = torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)

    return mask


def clip_frequency_magnitude(noise_magnitudes, clip_percentile=0.95):
    """Clip frequency domain magnitude to prevent large values."""

    # Calculate clipping threshold
    clip_threshold = torch.quantile(noise_magnitudes, clip_percentile)

    # Clip large values
    clipped_magnitudes = torch.clamp(noise_magnitudes, max=clip_threshold)

    return clipped_magnitudes


def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise: torch.Tensor = None,
    sampling_method: str = "fft",
) -> torch.Tensor:
    """
    Generate structured noise for a batch of images using frequency soft cutoff.
    Reduces boundary artifacts by padding images before FFT processing.

    Args:
        image_batch: Batch of image tensors of shape (B, C, H, W)
        noise_std: Standard deviation for Gaussian noise
        pad_factor: Padding factor (1.5 = 50% padding, 2.0 = 100% padding)
        cutoff_radius: Frequency cutoff radius (None = auto-calculate)
        transition_width: Width of smooth transition for frequency cutoff
        input_noise: Optional input noise tensor to use instead of generating new noise.
        sampling_method: Method to sample noise magnitude ('fft', 'cdf', 'two-gaussian')

    Returns:
        torch.Tensor: Batch of structured noise tensors of shape (B, C, H, W)
    """
    assert sampling_method in ["fft", "cdf", "two-gaussian"]
    # Ensure tensor is on the correct device
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()

    # Calculate padding size for overlap-add method
    pad_h = int(height * (pad_factor - 1))
    pad_h = pad_h // 2 * 2  # make it even
    pad_w = int(width * (pad_factor - 1))
    pad_w = pad_w // 2 * 2  # make it even

    # Pad images with reflection to reduce boundary artifacts
    padded_images = torch.nn.functional.pad(
        image_batch,
        (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
        mode="reflect",  # Mirror edges for natural transitions
    )

    # Calculate padded dimensions
    padded_height = height + pad_h
    padded_width = width + pad_w

    # Create frequency soft cutoff mask only if cutoff_radius is provided
    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height / 2, padded_width / 2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        # No cutoff - preserve all frequencies (full structure preservation)
        freq_mask = torch.ones(padded_height, padded_width, device=device)

    # Apply 2D FFT to padded images
    fft = torch.fft.fft2(padded_images, dim=(-2, -1))

    # Shift zero frequency to center
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Extract phase and magnitude for all images
    image_phases = torch.angle(fft_shifted)
    image_phases = clip_frequency_magnitude(image_phases)
    image_magnitudes = torch.abs(fft_shifted)

    if input_noise is not None:
        # Use provided noise
        noise_batch = torch.nn.functional.pad(
            input_noise,
            (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
            mode="reflect",  # Mirror edges for natural transitions
        )
        noise_batch = noise_batch.float()
    else:
        # Generate Gaussian noise for the padded size
        noise_batch = torch.randn_like(padded_images)

    # Extract noise magnitude and phase
    if sampling_method == "fft":
        # Apply 2D FFT to noise batch
        noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
        noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

        noise_magnitudes = torch.abs(noise_fft_shifted)
        noise_phases = torch.angle(noise_fft_shifted)
    elif sampling_method == "cdf":
        # The magnitude of FFT of Gaussian noise follows a Rayleigh distribution.
        # We can sample it directly.
        # The scale of the Rayleigh distribution is related to the std of the Gaussian noise
        # and the size of the FFT.
        # For an N-point FFT of Gaussian noise with variance sigma^2, the variance of
        # the real and imaginary parts of the FFT coefficients is N*sigma^2.
        # The scale parameter for the Rayleigh distribution is sqrt(N*sigma^2 / 2).
        # Here, N = padded_height * padded_width.

        N = padded_height * padded_width
        rayleigh_scale = (N / 2) ** 0.5

        ## Sample from a standard Rayleigh distribution (scale=1) and then scale it.
        uu = torch.rand(size=image_magnitudes.shape, device=device)
        noise_magnitudes = rayleigh_scale * torch.sqrt(-2.0 * torch.log(uu))
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

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
        # A standard Rayleigh can be generated from two standard normal distributions.
        u1 = torch.randn_like(image_magnitudes)
        u2 = torch.randn_like(image_magnitudes)
        noise_magnitudes = rayleigh_scale * torch.sqrt(u1**2 + u2**2)
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = (
                torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi
                - torch.pi
            )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    noise_magnitudes = clip_frequency_magnitude(noise_magnitudes)

    # Scale noise magnitude by standard deviation
    noise_magnitudes = noise_magnitudes * noise_std

    # Apply frequency soft cutoff to mix phases
    # Low frequencies (within cutoff) use image phase, high frequencies use noise phase
    mixed_phases = (
        freq_mask.unsqueeze(0).unsqueeze(0) * image_phases
        + (1 - freq_mask.unsqueeze(0).unsqueeze(0)) * noise_phases
    )

    # Combine magnitude and mixed phase for all images
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    # Shift zero frequency back to corner
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    # Apply inverse FFT
    structured_noise_padded = torch.fft.ifft2(fft_unshifted, dim=(-2, -1))
    # Take real part
    structured_noise_padded = torch.real(structured_noise_padded)

    clamp_mask = (structured_noise_padded < -5) + (structured_noise_padded > 5)
    clamp_mask = (clamp_mask > 0).float()

    structured_noise_padded = (
        structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    )

    # Crop back to original size (remove padding)
    structured_noise_batch = structured_noise_padded[
        :, :, pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width
    ]
    return structured_noise_batch.to(dtype)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16  # TRT engines expected to handle FP16
RESOLUTION = 512
NUM_STEPS = 4
IMAGE_INDEX = 170
SCALING_FACTOR = 1.0

# TRT Paths
TRT_ENCODER_PATH = "flux_vae_tiny_trt_v2/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt_v2/vae_decoder.plan"
TRT_UNET_PATH = "sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3/unet.plan"

OUTPUT_IMAGE_PATH = "inference_result_trt_full.png"


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
            raise FileNotFoundError(f"TRT Engine not found at {path}")
        with open(path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, feed_dict):
        # feed_dict maps input_name -> torch_tensor
        bindings = [None] * self.engine.num_io_tensors
        outputs = {}
        input_tensors = []  # keep alive until after synchronize()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)

            if mode == trt.TensorIOMode.INPUT:
                if tensor_name not in feed_dict:
                    print(
                        f"Warning: Missing input '{tensor_name}', expected: {feed_dict.keys()}"
                    )
                    raise ValueError(f"Missing input '{tensor_name}'")

                tensor = feed_dict[tensor_name]

                # Cast to the dtype this engine input actually declares
                _NP_TO_TORCH = {np.float16: torch.float16, np.float32: torch.float32}
                engine_np_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                torch_dtype = _NP_TO_TORCH.get(engine_np_dtype, torch.float32)
                if tensor.dtype != torch_dtype:
                    tensor = tensor.to(dtype=torch_dtype)

                # Ensure contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                input_tensors.append(tensor)  # prevent GC before execute
                self.context.set_input_shape(tensor_name, tensor.shape)
                self.context.set_tensor_address(tensor_name, tensor.data_ptr())
                bindings[i] = tensor.data_ptr()
            else:
                shape = self.context.get_tensor_shape(tensor_name)
                engine_np_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                torch_dtype = (
                    torch.float16 if engine_np_dtype == np.float16 else torch.float32
                )

                resolved_shape = list(shape)
                if -1 in resolved_shape:
                    batch_size = list(feed_dict.values())[0].shape[0]
                    if resolved_shape[0] == -1:
                        resolved_shape[0] = batch_size

                output_tensor = torch.empty(
                    tuple(resolved_shape), dtype=torch_dtype, device="cuda"
                )
                self.context.set_tensor_address(tensor_name, output_tensor.data_ptr())
                bindings[i] = output_tensor.data_ptr()
                outputs[tensor_name] = output_tensor

        # Wait for PyTorch default stream to finish producing inputs
        # (torch.cat, .to(), .contiguous() run on default stream; TRT runs on self.stream)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    print(f"Loading dataset item {IMAGE_INDEX}...")
    dataset_name = "dim/render_nfs_4screens_5_sdxl_1_wan_mix"
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    item = next(itertools.islice(dataset, IMAGE_INDEX, None))
    orig_source_pil = item["input_image"].convert("RGB")

    # 2. Load Engines
    print("Loading TensorRT Engines...")
    try:
        enc_engine = TRTEngine(TRT_ENCODER_PATH)
        dec_engine = TRTEngine(TRT_DECODER_PATH)
        unet_engine = TRTEngine(TRT_UNET_PATH)
    except Exception as e:
        print(f"Error loading engines: {e}")
        exit(1)

    print("Engines loaded.")

    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # 3. Preprocessing
    print("Preprocessing...")
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    c_t = train_transforms(orig_source_pil).unsqueeze(0).to(DEVICE).half()  # FP16 input

    # 4. Inference Pipeline
    print(f"Running Full TRT Pipeline ({NUM_STEPS} steps)...")

    sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
    noise_scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)

    # A. Encode
    print("  Encoding (TRT)...")
    enc_out = enc_engine.infer({"image": c_t})
    z_source_raw = list(enc_out.values())[0]
    # print(f"  [DEBUG] enc out  shape={z_source_raw.shape}  min={z_source_raw.min():.3f}  max={z_source_raw.max():.3f}")

    z_source = z_source_raw * SCALING_FACTOR

    # B. Diffusion Loop
    # Structured noise — mirrors flow_matching_inference_win.py generate_image()
    input_noise = torch.randn(z_source.shape, device=DEVICE, dtype=torch.float32)
    structured_noise = generate_structured_noise_batch_vectorized(
        z_source.float(),
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=100.0,
        input_noise=input_noise,
        sampling_method="fft",
    ).to(dtype=z_source.dtype, device=DEVICE)
    sample = structured_noise
    # print(f"  [DEBUG] noise    shape={sample.shape}  min={sample.min():.3f}  max={sample.max():.3f}")

    with torch.no_grad():
        for i, t in enumerate(noise_scheduler.timesteps):
            if hasattr(noise_scheduler, "scale_model_input"):
                denoiser_input = noise_scheduler.scale_model_input(sample, t)
            else:
                denoiser_input = sample

            # Concatenate: [denoised_sample, z_source] → [1, 64, 64, 64]
            cat_input = torch.cat([denoiser_input, z_source], dim=1)
            # print(f"  [step {i}] unet_in  shape={cat_input.shape}  min={cat_input.min():.3f}  max={cat_input.max():.3f}  t={t.item():.1f}")

            # Timestep — float32, shape [batch], mirrors trt_vae_torch_unet_inference.py
            t_tensor = t.to(DEVICE).repeat(cat_input.shape[0])

            # Infer UNet (TRT)
            unet_out = unet_engine.infer({"sample": cat_input, "timestep": t_tensor})
            pred = list(unet_out.values())[0]  # "out_sample"
            # print(f"  [step {i}] pred     shape={pred.shape}  min={pred.min():.3f}  max={pred.max():.3f}  nan={pred.isnan().any()}")

            # Scheduler Step
            sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]
            # print(f"  [step {i}] sample   shape={sample.shape}  min={sample.min():.3f}  max={sample.max():.3f}")

    # C. Decode
    print("  Decoding (TRT)...")
    latents_to_decode = sample / SCALING_FACTOR
    # print(f"  [DEBUG] dec_in   shape={latents_to_decode.shape}  min={latents_to_decode.min():.3f}  max={latents_to_decode.max():.3f}")
    dec_out = dec_engine.infer({"latent": latents_to_decode})
    output_image = list(dec_out.values())[0]
    # print(f"  [DEBUG] dec_out  shape={output_image.shape}  min={output_image.min():.3f}  max={output_image.max():.3f}  nan={output_image.isnan().any()}")

    # Post-process
    output_image = output_image.clamp(-1, 1)

    # Check shape for PIL
    if output_image.dim() == 4:
        image_tensor = output_image[0]
    else:
        image_tensor = output_image

    print(f"  Decoder Output Shape: {output_image.shape}")

    output_pil = transforms.ToPILImage()(image_tensor.cpu().float() * 0.5 + 0.5)

    print(f"Saving to {OUTPUT_IMAGE_PATH}")
    output_pil.save(OUTPUT_IMAGE_PATH)
    print("Success.")
