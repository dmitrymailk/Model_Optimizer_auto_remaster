
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import onnx
import tensorrt as trt
from diffusers import UNet2DModel
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize UNet Model")
    parser.add_argument("--output-dir", type=str, default="unet_trt", help="Output directory for ONNX and TRT engine")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp32", action="store_true", help="Export in FP32 (default is FP16)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to pretrained model checkpoint")
    return parser.parse_args()

def get_unet_model(device, dtype, model_path=None):
    if model_path:
        logger.info(f"Loading UNet2DModel from {model_path}...")
        unet = UNet2DModel.from_pretrained(
            model_path, 
            subfolder="unet", 
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)
    else:
        # Configuration from unet_and_vae.ipynb
        # unet2d_config = {
        #     "sample_size": 32,
        #     "in_channels": 128,
        #     "out_channels": 128,
        #     "center_input_sample": False,
        #     "time_embedding_type": "positional",
        #     "freq_shift": 0,
        #     "flip_sin_to_cos": True,
        #     "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        #     "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        #     "block_out_channels": [320, 640, 1280],
        #     "layers_per_block": 1,
        #     "mid_block_scale_factor": 1,
        #     "downsample_padding": 1,
        #     "downsample_type": "conv",
        #     "upsample_type": "conv",
        #     "dropout": 0.0,
        #     "act_fn": "silu",
        #     "norm_num_groups": 32,
        #     "norm_eps": 1e-05,
        #     "resnet_time_scale_shift": "default",
        #     "add_attention": False,
        # }
        
        # logger.info("Creating UNet2DModel with configuration (Random Init)...")
        # unet = UNet2DModel(**unet2d_config).to(device, dtype=dtype)

        logger.error(
            "No model path provided and no default configuration is active.\n"
            "  Please specify a checkpoint via --model-path, e.g.:\n"
            "    python optimize_unet.py --model-path <path/to/checkpoint>"
        )
        sys.exit(1)

    unet.requires_grad_(False)
    unet.eval()
    return unet

class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, sample, timestep):
        # UNet forward returns a UNet2DOutput object, accessing [0] gives the sample
        return self.unet(sample, timestep, return_dict=False)[0]

def export_onnx(unet, output_path, opset=17, fp16=False):
    logger.info(f"Exporting UNet to ONNX at {output_path}...")
    
    # Input shapes from notebook:
    # latents torch.Size([1, 128, 32, 32])
    # t (timestep) - scalar or batch size 1
    
    
    B = 1
    # Use config from loaded model if available
    if hasattr(unet, "config"):
        C = unet.config.in_channels
        H = unet.config.sample_size
        W = unet.config.sample_size
        logger.info(f"Using input shape from config: B={B}, C={C}, H={H}, W={W}")
    else:
        # Fallback
        C = 256
        H = 32
        W = 32
        logger.info(f"Using fallback input shape: B={B}, C={C}, H={H}, W={W}")
    
    device = "cuda" if fp16 else "cpu"
    dtype = torch.float16 if fp16 else torch.float32
    
    dummy_sample = torch.randn(B, C, H, W, device=device, dtype=dtype)
    dummy_timestep = torch.tensor([1.0], device=device, dtype=dtype) # Timestep is usually float or long, diffusers handles cast
    # Note: Diffusers UNet usually expects timestep as tensor of shape (B,) or just a scalar.
    
    model_wrapper = UNetWrapper(unet)

    torch.onnx.export(
        model_wrapper,
        (dummy_sample, dummy_timestep),
        output_path,
        input_names=["sample", "timestep"],
        output_names=["out_sample"],
        dynamic_axes=None, # Static shape as requested for optimization
        opset_version=opset,
        do_constant_folding=True
    )
    logger.info("ONNX export successful.")

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
    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX file")
            
    # Optimization inputs
    profile = builder.create_optimization_profile()
    
    input_tensor = network.get_input(0) # sample
    input_name = input_tensor.name
    
    # Infer shape from input tensor if possible
    # Dimensions: (batch_size, channels, height, width)
    in_dims = input_tensor.shape
    logger.info(f"ONNX Input Dims: {in_dims}")

    # Use dimensions from ONNX if they are concrete (>0), else fallback
    # Assuming B=1, static export
    C = in_dims[1] if in_dims[1] > 0 else 128
    H = in_dims[2] if in_dims[2] > 0 else 32
    W = in_dims[3] if in_dims[3] > 0 else 32
    
    profile.set_shape(input_name, (1, C, H, W), (1, C, H, W), (1, C, H, W))
    
    # Also set shape for timestep if it's an input in the network definition
    if network.num_inputs > 1:
        timestep_tensor = network.get_input(1)
        timestep_name = timestep_tensor.name
        # Assuming scalar or B=1
        profile.set_shape(timestep_name, (1,), (1,), (1,))

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
    
    onnx_path = os.path.join(args.output_dir, "unet.onnx")
    engine_path = os.path.join(args.output_dir, "unet.plan")
    
    use_fp16 = not args.fp32
    
    device = "cuda" if use_fp16 else "cpu"
    dtype = torch.float16 if use_fp16 else torch.float32
    
    if use_fp16 and not torch.cuda.is_available():
         logger.warning("FP16 requested but CUDA not available. Switching to CPU/FP32 for export (TRT build will fail if no GPU).")
         device = "cpu"
         dtype = torch.float32
         use_fp16 = False
         
    unet = get_unet_model(device, dtype, model_path=args.model_path)
    
    export_onnx(unet, onnx_path, opset=args.opset, fp16=use_fp16)
    
    if os.path.exists(onnx_path):
        # TRT build requires GPU
        if torch.cuda.is_available():
            build_trt_engine(onnx_path, engine_path, fp16=use_fp16)
        else:
             logger.warning("Skipping TensorRT build because CUDA is not available.")
    else:
        logger.error("ONNX export failed.")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
