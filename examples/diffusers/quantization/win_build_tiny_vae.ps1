$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run optimization for Tiny AutoEncoder
# Using FP16 and explicit latent-dim 32 (512px / 16 downscale)
# . "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python optimize_flux_vae.py --model-path "fal/FLUX.2-Tiny-AutoEncoder" --output-dir "flux_vae_tiny_trt" --fp16 --latent-dim 32
. "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python optimize_flux_vae.py --model-path "dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora" --output-dir "flux_vae_tiny_trt_v2" --fp16 --latent-dim 32 --with-pixel-shuffle
