import argparse
import logging
import os
import time
import sys

import torch
import tensorrt as trt
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, AutoModel
from torchvision.utils import save_image

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Flux VAE: Torch vs TensorRT"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="fal/FLUX.2-Tiny-AutoEncoder",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        default="flux_vae_tiny_trt/vae_decoder.plan",
        help="Path to TensorRT Decoder engine",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="flux_vae_tiny_trt/vae_encoder.plan",
        help="Path to TensorRT Encoder engine",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use FP32 for benchmark (default: False, uses FP16)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=r"C:\programming\auto_remaster\inference_optimization\170_2x.png",
        help="Path to input image for benchmarking",
    )
    parser.add_argument(
        "--with-pixel-shuffle",
        action="store_true",
        help="Apply pixel_shuffle(2) after encode and pixel_unshuffle(2) before decode. "
             "Must match what was baked into the TRT engines.",
    )
    parser.add_argument(
        "--pixel-shuffle-factor",
        type=int,
        default=2,
        help="pixel_shuffle / pixel_unshuffle upscale factor (default: 2)",
    )
    return parser.parse_args()


def process_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")

    # Resize ensuring shortest edge is 'size', then center crop
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Map to [-1, 1]
        ]
    )

    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0)  # Add batch dim -> (1, 3, H, W)
    return pixel_values


def save_output_image(tensor, filename):
    # Tensor is (1, 3, H, W) in [-1, 1] range
    # Denormalize to [0, 1], force float32 for correct save
    t = tensor.float()
    logger.info(f"[SAVE {filename}]  min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")
    image = (t * 0.5 + 0.5).clamp(0, 1)
    save_image(image, filename)
    logger.info(f"Saved reconstructed image to {filename}")


def benchmark_pipeline(name, func, warmth=10, runs=100, cuda_stream=None):
    # Warmup
    for _ in range(warmth):
        with torch.no_grad():
            output = func()
    torch.cuda.synchronize()

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(cuda_stream)
    for _ in range(runs):
        with torch.no_grad():
            output = func()
    end_event.record(cuda_stream)
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / runs
    fps = 1000.0 / avg_time_ms
    logger.info(f"{name}: {avg_time_ms:.2f} ms ({fps:.1f} FPS)")
    
    return avg_time_ms, fps, output


def benchmark_torch(vae, pixel_values, warmth=10, runs=100,
                    with_pixel_shuffle=False, pixel_shuffle_factor=2):
    logger.info(f"Benchmarking Torch Full Pipeline ({pixel_values.dtype}){' + pixel_shuffle' if with_pixel_shuffle else ''}...")

    def run_pipeline():
        # Encode
        encoded_output = vae.encode(pixel_values)
        if hasattr(encoded_output, "latent"):
            # Flux2TinyAutoEncoder returns EncoderOutput with .latent
            latents = encoded_output.latent
        elif hasattr(encoded_output, "latent_dist"):
            latents = encoded_output.latent_dist.sample()
        elif hasattr(encoded_output, "latents"):
            latents = encoded_output.latents
        else:
            latents = encoded_output[0]

        # Apply pixel_shuffle to match TRT encoder output
        if with_pixel_shuffle:
            latents = torch.nn.functional.pixel_shuffle(latents, pixel_shuffle_factor)

        # Apply pixel_unshuffle to match TRT decoder input expectation
        if with_pixel_shuffle:
            latents = torch.nn.functional.pixel_unshuffle(latents, pixel_shuffle_factor)

        # Decode
        decoded = vae.decode(latents)
        if hasattr(decoded, "sample"):
            output = decoded.sample
        elif isinstance(decoded, (tuple, list)):
            output = decoded[0]
        else:
            output = decoded
        return output

    return benchmark_pipeline("Torch Full Pipeline", run_pipeline, warmth, runs)



def load_engine(engine_path):
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found: {engine_path}")
        return None
    logger.info(f"Loading engine from {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def benchmark_trt(encoder_path, decoder_path, pixel_values, warmth=10, runs=100):
    logger.info("Benchmarking TensorRT Full Pipeline...")

    enc_engine = load_engine(encoder_path)
    dec_engine = load_engine(decoder_path)

    if not enc_engine or not dec_engine:
        logger.error("Both Encoder and Decoder engines are required.")
        return None, None, None

    enc_context = enc_engine.create_execution_context()
    dec_context = dec_engine.create_execution_context()

    stream = torch.cuda.Stream()
    
    # --- Prepare Buffers ---
    # 1. Encoder Input (Image)
    enc_input_name = "image"
    enc_output_name = "latent"
    for i in range(enc_engine.num_io_tensors):
        name = enc_engine.get_tensor_name(i)
        if enc_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: enc_input_name = name
        elif enc_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT: enc_output_name = name

    enc_context.set_input_shape(enc_input_name, pixel_values.shape)
    
    # 2. Latents (Enc Output / Dec Input)
    # Infer latent shape from encoder output shape
    enc_out_shape = enc_engine.get_tensor_shape(enc_output_name)
    # Fix dynamic batch if present
    latent_shape = (pixel_values.shape[0],) + tuple(enc_out_shape[1:])
    latents_tensor = torch.zeros(latent_shape, dtype=torch.float16, device="cuda")
    
    # 3. Decoder Output (Image)
    dec_input_name = "latent"
    dec_output_name = "image"
    for i in range(dec_engine.num_io_tensors):
        name = dec_engine.get_tensor_name(i)
        if dec_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: dec_input_name = name
        elif dec_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT: dec_output_name = name
        
    dec_context.set_input_shape(dec_input_name, latent_shape)
    
    # Infer output shape
    dec_out_shape = dec_engine.get_tensor_shape(dec_output_name)
    # logger.info(f"Engine Output Shape: {dec_out_shape}") # Optional debug
    
    if len(dec_out_shape) == 3:
         # (C, H, W) -> (B, C, H, W)
         output_shape = (pixel_values.shape[0],) + tuple(dec_out_shape)
    else:
         # (B, C, H, W) or (-1, C, H, W)
         output_shape = (pixel_values.shape[0],) + tuple(dec_out_shape[1:])
    
    # If dynamic, we might need manual calculation. But here we expect fixed.
    if -1 in output_shape:
         output_shape = (pixel_values.shape[0], 3, pixel_values.shape[2], pixel_values.shape[3])

    output_tensor = torch.zeros(output_shape, dtype=torch.float16, device="cuda")

    # --- Set Addresses ---
    enc_context.set_tensor_address(enc_input_name, pixel_values.data_ptr())
    enc_context.set_tensor_address(enc_output_name, latents_tensor.data_ptr())
    dec_context.set_tensor_address(dec_input_name, latents_tensor.data_ptr())
    dec_context.set_tensor_address(dec_output_name, output_tensor.data_ptr())

    def run_full_trt():
        enc_context.execute_async_v3(stream_handle=stream.cuda_stream)
        dec_context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return output_tensor

    return benchmark_pipeline("TRT Full Pipeline", run_full_trt, warmth, runs, cuda_stream=stream)


def main():
    args = parse_args()

    # Load VAE
    if "Tiny-AutoEncoder" in args.model_path:
        logger.info("Detected Tiny AutoEncoder.")
        vae = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")

    dtype = torch.float32 if args.float32 else torch.float16
    vae.to(device="cuda", dtype=dtype)
    vae.eval()

    # Process Image
    logger.info(f"Processing image from {args.image_path}...")
    pixel_values = process_image(args.image_path, size=512)
    pixel_values = pixel_values.to(device="cuda", dtype=dtype)

    # Run Torch Benchmark
    torch_ms, torch_fps, torch_out = benchmark_torch(
        vae, pixel_values,
        with_pixel_shuffle=args.with_pixel_shuffle,
        pixel_shuffle_factor=args.pixel_shuffle_factor,
    )
    save_output_image(torch_out, "vae_output_torch.png")
    
    print("-" * 50)

    # Run TRT Benchmark
    if os.path.exists(args.encoder_path) and os.path.exists(args.decoder_path):
        trt_ms, trt_fps, trt_out = benchmark_trt(
            args.encoder_path, args.decoder_path, pixel_values
        )
        if trt_out is not None:
             save_output_image(trt_out, "vae_output_trt.png")
        
        print("\n" + "=" * 50)
        print("   VAE FULL PIPELINE BENCHMARK")
        print("=" * 50)
        print(f"Torch:      {torch_ms:6.2f} ms  ({torch_fps:6.1f} FPS)")
        print(f"TensorRT:   {trt_ms:6.2f} ms  ({trt_fps:6.1f} FPS)")
        print("-" * 50)
        print(f"Speedup:    {torch_ms/trt_ms:6.2f}x")
        print("=" * 50 + "\n")
    else:
        print("TRT Engines not found. Skipping.")


if __name__ == "__main__":
    main()
