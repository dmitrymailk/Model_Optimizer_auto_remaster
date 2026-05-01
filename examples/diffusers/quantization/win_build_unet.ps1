$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run optimization for UNet
# Model Path: checkpoint-299200
# $ModelPath = "C:\programming\auto_remaster\inference_optimization\models\lbm_train_test_gap_tiny_v6_upscale_2x\checkpoint-128000"
$ModelPath = "C:\programming\auto_remaster\inference_optimization\models\sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3\student"
$OutputDir = "sid_klein_lora_gan_patch_lpips_sid_anchor_20x_v3"

# Clean output directory
if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}

. "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python optimize_unet.py --model-path $ModelPath --output-dir $OutputDir --opset 18 --fp16
