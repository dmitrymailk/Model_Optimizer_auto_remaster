
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import onnx
import tensorrt as trt
from diffusers import FluxPipeline
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Flux VAE Decoder")
    parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-dev", help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="flux_vae_trt", help="Output directory for ONNX and TRT engine")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for optimization profile")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension (H=W) for export. Default 64 (512px image / 8). Use 32 for Tiny AutoEncoder on 512px.")
    parser.add_argument("--with-pixel-shuffle", action="store_true", help="Apply pixel_shuffle after encoding (for Tiny AutoEncoder with upscale factor). Reduces channel count by factor^2, increases spatial dims.")
    parser.add_argument("--pixel-shuffle-factor", type=int, default=2, help="Upscale factor for pixel_shuffle (default: 2). Channels become C/factor^2, spatial dims become H*factor x W*factor.")
    return parser.parse_args()

def export_onnx(
    vae,
    output_path,
    opset=17,
    fp16=False,
    channels=16,
    latent_dim=64,
    with_pixel_shuffle=False,
    pixel_shuffle_factor=2,
):
    logger.info(f"Exporting VAE Decoder to ONNX at {output_path}...")
    
    # When pixel_shuffle is enabled the decoder receives the pixel_shuffled latent:
    #   channels_in = channels / factor^2,  spatial_dim_in = latent_dim * factor
    # and applies pixel_unshuffle inside before calling vae.decode().
    if with_pixel_shuffle:
        dec_channels = channels // (pixel_shuffle_factor ** 2)
        dec_spatial  = latent_dim * pixel_shuffle_factor
        logger.info(
            f"  pixel_unshuffle enabled (factor={pixel_shuffle_factor}): "
            f"decoder input [{dec_channels}, {dec_spatial}, {dec_spatial}] "
            f"→ pixel_unshuffle → [{channels}, {latent_dim}, {latent_dim}]"
        )
    else:
        dec_channels = channels
        dec_spatial  = latent_dim

    B = 1
    C = dec_channels
    H = dec_spatial
    W = dec_spatial
    
    device = "cuda" if fp16 else "cpu"
    dtype = torch.float16 if fp16 else torch.float32
    vae.to(device=device, dtype=dtype)
    vae.eval()
    
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
    
    # Wrapper to only call decoder
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae, with_pixel_shuffle, pixel_shuffle_factor):
            super().__init__()
            self.vae = vae
            self.with_pixel_shuffle = with_pixel_shuffle
            self.pixel_shuffle_factor = pixel_shuffle_factor
            
        def forward(self, x):
            # Mirror Flux2TinyAutoEncoder.decode() from flow_matching_inference_win.py.
            # Use manual reshape+permute instead of pixel_unshuffle to avoid TRT axis bugs.
            # pixel_unshuffle(x, r): [B, C, H*r, W*r] -> [B, C*r^2, H, W]
            if self.with_pixel_shuffle:
                r  = self.pixel_shuffle_factor
                B, C, Hf, Wf = x.shape          # [1, 32, 64, 64]
                H, W = Hf // r, Wf // r          # 32, 32
                # Step 1: reshape to split spatial into blocks
                x = x.reshape(B, C, H, r, W, r)  # [1, 32, 32, 2, 32, 2]
                # Step 2: bring block dims next to channel
                x = x.permute(0, 1, 3, 5, 2, 4)  # [1, 32, 2, 2, 32, 32]
                # Step 3: merge channels + blocks -> new channel dim
                x = x.reshape(B, C * r * r, H, W) # [1, 128, 32, 32]
            out = self.vae.decode(x).sample
            return out

    model_wrapper = VAEDecoderWrapper(vae, with_pixel_shuffle, pixel_shuffle_factor)

    torch.onnx.export(
        model_wrapper,
        (dummy_input,),
        output_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True
    )
    logger.info("ONNX export successful.")

def export_encoder_onnx(
    vae,
    output_path,
    opset=17,
    fp16=False,
    image_size=512,
    with_pixel_shuffle=False,
    pixel_shuffle_factor=2,
):
    logger.info(f"Exporting VAE Encoder to ONNX at {output_path}...")
    if with_pixel_shuffle:
        logger.info(f"  pixel_shuffle enabled (factor={pixel_shuffle_factor}): channels /= {pixel_shuffle_factor**2}, spatial *= {pixel_shuffle_factor}")
    
    B = 1
    C = 3
    H = image_size
    W = image_size
    
    device = "cuda" if fp16 else "cpu"
    dtype = torch.float16 if fp16 else torch.float32
    vae.to(device=device, dtype=dtype)
    vae.eval()
    
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
    
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae, with_pixel_shuffle, pixel_shuffle_factor):
            super().__init__()
            self.vae = vae
            self.with_pixel_shuffle = with_pixel_shuffle
            self.pixel_shuffle_factor = pixel_shuffle_factor
            
        def forward(self, x):
            # Mirror Flux2TinyAutoEncoder.encode() from flow_matching_inference_win.py:
            #   pixel_shuffle(self.vae.encode(x).latent, 2)
            latents = self.vae.encode(x).latent
            if self.with_pixel_shuffle:
                latents = torch.nn.functional.pixel_shuffle(latents, self.pixel_shuffle_factor)
            return latents

    model_wrapper = VAEEncoderWrapper(vae, with_pixel_shuffle, pixel_shuffle_factor)

    torch.onnx.export(
        model_wrapper,
        (dummy_input,),
        output_path,
        input_names=["image"],
        output_names=["latent"],
        dynamic_axes=None,  # Static shapes for now
        opset_version=opset,
        do_constant_folding=True
    )
    logger.info("Encoder ONNX export successful.")

def build_trt_engine(onnx_path, engine_path, fp16=False, verbose=False):
    logger.info(f"Building TensorRT engine at {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform does not support fast FP16, falling back to FP32.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX
    # Use parse_from_file to correctly handle external data files
    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX file")
            
    # Optimization inputs
    profile = builder.create_optimization_profile()
    
    input_tensor = network.get_input(0)
    input_name = input_tensor.name if input_tensor else "input"
    
    if input_tensor:
        in_dims = input_tensor.shape
        logger.info(f"Parsed ONNX input dims for '{input_name}': {in_dims}")
        
        c_val = in_dims[1] if in_dims[1] > 0 else 16
        h_val = in_dims[2] if in_dims[2] > 0 else 64
        w_val = in_dims[3] if in_dims[3] > 0 else 64
        
        # B=1 for this specific task
        profile.set_shape(input_name, (1, c_val, h_val, w_val), (1, c_val, h_val, w_val), (1, c_val, h_val, w_val))
        
        # DEBUG: Print output info
        for i in range(network.num_outputs):
            out_tensor = network.get_output(i)
            logger.info(f"Parsed ONNX OUTPUT {i} '{out_tensor.name}' dims: {out_tensor.shape}")

    else:
        # Fallback
        logger.warning(f"Could not determine input dims, using fallback for '{input_name}'")
        profile.set_shape(input_name, (1, 16, 64, 64), (1, 16, 64, 64), (1, 16, 64, 64))
        
    config.add_optimization_profile(profile)
    
    # Build
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Engine build failed")
            
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info("TensorRT engine build successful.")
        
    except Exception as e:
        logger.error(f"Error building engine: {e}")
        raise

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for Decoder
    dec_onnx_path = os.path.join(args.output_dir, "vae_decoder.onnx")
    dec_engine_path = os.path.join(args.output_dir, "vae_decoder.plan")
    
    # Paths for Encoder
    enc_onnx_path = os.path.join(args.output_dir, "vae_encoder.onnx")
    enc_engine_path = os.path.join(args.output_dir, "vae_encoder.plan")
    
    logger.info(f"Loading Flux Pipeline from {args.model_path}...")
    try:
        from diffusers import AutoencoderKL, AutoModel
        
        # Check for Tiny AutoEncoder
        if "Tiny-AutoEncoder" in args.model_path:
            logger.info("Detected Tiny AutoEncoder. Loading with trust_remote_code=True...")
            vae = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        else:
            subfolder = "vae"
            try:
                vae = AutoencoderKL.from_pretrained(args.model_path, subfolder=subfolder)
            except:
                logger.info("Could not load with subfolder='vae', trying direct load...")
                vae = AutoencoderKL.from_pretrained(args.model_path)
            
    except Exception as e:
        logger.error(f"Failed to load VAE: {e}")
        sys.exit(1)
        
    # Debug: Inspect VAE
    logger.info(f"VAE Config: {vae.config}")
    
    # Try to determine correct channels from config
    channels = 16 # Default Flux
    if hasattr(vae.config, "latent_channels"):
        channels = vae.config.latent_channels
    elif hasattr(vae.config, "in_channels"):
        channels = vae.config.in_channels
        
    logger.info(f"Using {channels} channels for export.")
    
    # --- ENCODER ---
    logger.info("--- Processing Encoder ---")
    export_encoder_onnx(
        vae,
        enc_onnx_path,
        opset=args.opset,
        fp16=args.fp16,
        image_size=512,
        with_pixel_shuffle=args.with_pixel_shuffle,
        pixel_shuffle_factor=args.pixel_shuffle_factor,
    )
    build_trt_engine(enc_onnx_path, enc_engine_path, fp16=args.fp16)

    # --- DECODER ---
    logger.info("--- Processing Decoder ---")
    export_onnx(
        vae,
        dec_onnx_path,
        opset=args.opset,
        fp16=args.fp16,
        channels=channels,
        latent_dim=args.latent_dim,
        with_pixel_shuffle=args.with_pixel_shuffle,
        pixel_shuffle_factor=args.pixel_shuffle_factor,
    )
    build_trt_engine(dec_onnx_path, dec_engine_path, fp16=args.fp16)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
