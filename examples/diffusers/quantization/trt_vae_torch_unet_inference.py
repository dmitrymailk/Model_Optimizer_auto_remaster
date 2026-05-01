
import os
import torch
import numpy as np
import tensorrt as trt
import itertools
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import FlowMatchEulerDiscreteScheduler, UNet2DModel


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
    mask = torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)
    return mask


def clip_frequency_magnitude(noise_magnitudes, clip_percentile=0.95):
    clip_threshold = torch.quantile(noise_magnitudes, clip_percentile)
    return torch.clamp(noise_magnitudes, max=clip_threshold)


def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise: torch.Tensor = None,
    sampling_method: str = "fft",
) -> torch.Tensor:
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()
    pad_h = int(height * (pad_factor - 1)) // 2 * 2
    pad_w = int(width  * (pad_factor - 1)) // 2 * 2
    padded_images = torch.nn.functional.pad(
        image_batch, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), mode="reflect"
    )
    padded_height = height + pad_h
    padded_width  = width  + pad_w
    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height / 2, padded_width / 2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        freq_mask = torch.ones(padded_height, padded_width, device=device)
    fft_shifted = torch.fft.fftshift(torch.fft.fft2(padded_images, dim=(-2, -1)), dim=(-2, -1))
    image_phases = clip_frequency_magnitude(torch.angle(fft_shifted))
    image_magnitudes = torch.abs(fft_shifted)
    if input_noise is not None:
        noise_batch = torch.nn.functional.pad(
            input_noise, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), mode="reflect"
        ).float()
    else:
        noise_batch = torch.randn_like(padded_images)
    noise_fft_shifted = torch.fft.fftshift(torch.fft.fft2(noise_batch, dim=(-2, -1)), dim=(-2, -1))
    noise_magnitudes = clip_frequency_magnitude(torch.abs(noise_fft_shifted)) * noise_std
    noise_phases = torch.angle(noise_fft_shifted)
    fm = freq_mask.unsqueeze(0).unsqueeze(0)
    mixed_phases = fm * image_phases + (1 - fm) * noise_phases
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    structured_noise_padded = torch.real(
        torch.fft.ifft2(torch.fft.ifftshift(fft_combined, dim=(-2, -1)), dim=(-2, -1))
    )
    clamp_mask = ((structured_noise_padded < -5) | (structured_noise_padded > 5)).float()
    structured_noise_padded = structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    return structured_noise_padded[
        :, :, pad_h // 2: pad_h // 2 + height, pad_w // 2: pad_w // 2 + width
    ].to(dtype)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16
RESOLUTION = 512
NUM_STEPS = 1
IMAGE_INDEX = 170
SCALING_FACTOR = 1.0  # Flux2TinyAutoEncoder.config.scaling_factor

CHECKPOINT_PATH = r"c:\programming\auto_remaster\inference_optimization\models\sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3\student"
TRT_ENCODER_PATH = "flux_vae_tiny_trt_v2/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt_v2/vae_decoder.plan"
OUTPUT_IMAGE_PATH = "inference_result_trt_vae.png"

# -----------------------------------------------------------------------------
# TensorRT Utilities
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
        # Returns dict output_name -> torch_tensor
        
        bindings = [None] * self.engine.num_io_tensors
        outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            if mode == trt.TensorIOMode.INPUT:
                if tensor_name not in feed_dict:
                    raise ValueError(f"Missing input '{tensor_name}'")
                tensor = feed_dict[tensor_name]
                self.context.set_input_shape(tensor_name, tensor.shape)
                self.context.set_tensor_address(tensor_name, tensor.data_ptr())
                bindings[i] = tensor.data_ptr()
            else:
                # Output
                shape = self.context.get_tensor_shape(tensor_name)
                # Check for dynamic dims (-1)
                # Try to infer from input batch size if needed, assuming B is dim 0
                # For this specific VAE, shapes are likely fixed or B,C,H,W
                # If shape has -1, we might need logic to deduce it.
                # Assuming static shapes for now based on export script (1, C, H, W)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                # Map numpy dtype to torch dtype
                if dtype == np.float16: torch_dtype = torch.float16
                elif dtype == np.float32: torch_dtype = torch.float32
                else: torch_dtype = torch.float32 # Fallback
                
                # Allocation
                # If dynamic, we must resolve shape. 
                # For encoder: output latent (1, 16, 64, 64) likely
                # For decoder: output image (1, 3, 512, 512) likely
                
                resolved_shape = list(shape)
                if -1 in resolved_shape:
                     # Simple heuristic: use batch size from first input
                     batch_size = list(feed_dict.values())[0].shape[0]
                     resolved_shape[0] = batch_size
                
                output_tensor = torch.empty(tuple(resolved_shape), dtype=torch_dtype, device="cuda")
                self.context.set_tensor_address(tensor_name, output_tensor.data_ptr())
                bindings[i] = output_tensor.data_ptr()
                outputs[tensor_name] = output_tensor

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    print(f"Loading dataset item {IMAGE_INDEX}...")
    dataset_name = "dim/render_nfs_4screens_5_sdxl_1_wan_mix"
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    item = next(itertools.islice(dataset, IMAGE_INDEX, None))
    orig_source_pil = item["input_image"].convert("RGB")
    
    # 2. Load Models
    print("Loading TensorRT VAE Engines...")
    try:
        enc_engine = TRTEngine(TRT_ENCODER_PATH)
        dec_engine = TRTEngine(TRT_DECODER_PATH)
    except Exception as e:
        print(f"Error loading TRT engines: {e}")
        print("Please ensure 'flux_vae_tiny_trt/vae_encoder.plan' and 'flux_vae_tiny_trt/vae_decoder.plan' exist.")
        exit(1)
        
    print(f"Loading Torch UNet form {CHECKPOINT_PATH}...")
    unet = UNet2DModel.from_pretrained(
        CHECKPOINT_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        use_safetensors=True
    ).to(DEVICE)
    unet.eval()
    
    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 3. Preprocessing
    print("Preprocessing...")
    train_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Input tensor (1, 3, 512, 512)
    # The VAE TRT engine likely expects FP16 if built with --fp16
    # We should check dtype required. Usually inputs match engine precision if strong typing was used.
    # Convert to half for FP16 engine
    c_t = train_transforms(orig_source_pil).unsqueeze(0).to(DEVICE).half()
    
    # 4. Inference
    print(f"Running inference with {NUM_STEPS} steps...")
    
    sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
    noise_scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)
    
    # --- VAE ENCODE (TRT) ---
    print("Encoding with TRT...")
    enc_outputs = enc_engine.infer({"image": c_t})
    
    # Output name likely "latent"
    z_source_raw = list(enc_outputs.values())[0] # Take first output
    print(f"TRT Encoder Output Shape: {z_source_raw.shape}")
    print(f"TRT Encoder Output Stats: min={z_source_raw.min():.4f}, max={z_source_raw.max():.4f}, mean={z_source_raw.mean():.4f}")
    
    # --- Scaling ---
    z_source = z_source_raw * SCALING_FACTOR
    
    # --- LOOP ---
    # Structured noise init — mirrors flow_matching_inference_win.py
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
    
    with torch.no_grad():
        for i, t in enumerate(noise_scheduler.timesteps):
            if hasattr(noise_scheduler, "scale_model_input"):
                 denoiser_input = noise_scheduler.scale_model_input(sample, t)
            else:
                 denoiser_input = sample
                 
            # Concatenate
            unet_input = torch.cat([denoiser_input, z_source], dim=1)
            t_batch = t.to(DEVICE).repeat(unet_input.shape[0])
            
            # UNet Forward (Torch)
            pred = unet(unet_input, t_batch, return_dict=False)[0]
            
            # Step
            sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]
            
    # --- VAE DECODE (TRT) ---
    print("Decoding with TRT...")
    # Unscale
    latents_to_decode = sample / SCALING_FACTOR
    print(f"TRT Decoder Input Shape: {latents_to_decode.shape}")
    
    # Input name likely "latent"
    dec_outputs = dec_engine.infer({"latent": latents_to_decode})
    output_image = list(dec_outputs.values())[0]
    print(f"TRT Decoder Output Shape: {output_image.shape}")
    print(f"TRT Decoder Output Stats: min={output_image.min():.4f}, max={output_image.max():.4f}, mean={output_image.mean():.4f}")

        
    # Post-process
    output_image = output_image.clamp(-1, 1)
    
    # Check if we have batch dimension
    if output_image.dim() == 4:
        # (B, C, H, W) -> take first
        image_tensor = output_image[0]
    else:
        # (C, H, W) -> use as is
        image_tensor = output_image
        
    output_pil = transforms.ToPILImage()(image_tensor.cpu().float() * 0.5 + 0.5)
        
    print(f"Saving result to {OUTPUT_IMAGE_PATH}")
    output_pil.save(OUTPUT_IMAGE_PATH)
    print("Done.")
