# ComfyUI diffusers wrapper nodes for [Framer](https://aim-uofa.github.io/Framer/)

Framer models loaded from `ComfyUI/models/diffusion_models`

https://huggingface.co/Kijai/Framer_comfy

The VAE and image encoder from diffusers SVD-XT 1.1 is currently also required and autodownloaded to:

```
ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1
│   model_index.json
│
├───feature_extractor
│       preprocessor_config.json
│
├───image_encoder
│       config.json
│       model.fp16.safetensors
│
├───scheduler
│       scheduler_config.json
│
└───vae
        config.json
        diffusion_pytorch_model.fp16.safetensors
```