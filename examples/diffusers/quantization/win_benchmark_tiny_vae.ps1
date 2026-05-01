$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run benchmark
# . "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python benchmark_vae.py --model-path "fal/FLUX.2-Tiny-AutoEncoder" --decoder-path "flux_vae_tiny_trt/vae_decoder.plan" --encoder-path "flux_vae_tiny_trt/vae_encoder.plan"
. "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python benchmark_vae.py --model-path "dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora_v2" --decoder-path "flux_vae_tiny_trt_v2/vae_decoder.plan" --encoder-path "flux_vae_tiny_trt_v2/vae_encoder.plan" --with-pixel-shuffle
