import os
import torch
import json
import gc
from .utils import log, print_memory, get_vis_image, interpolate_trajectory
from diffusers.video_processor import VideoProcessor
from typing import List, Dict, Any, Tuple

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from .models_diffusers.controlnet_svd import ControlNetSVDModel
from .models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline

import folder_paths
folder_paths.add_model_folder_path("hyvid_embeds", os.path.join(folder_paths.get_output_directory(), "hyvid_embeds"))

import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

#region Model loading
class FramerModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                "controlnet": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "fp16"], {"default": "fp16"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "xformers",
                    ], {"default": "sdpa"}),
                "compile_args": ("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("FRAMERMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramerWrapper"

    def loadmodel(self, unet, controlnet, base_precision, load_device, compile_args=None, attention_mode="sdpa"):
        transformer = None
        mm.unload_all_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        # UNET
        framer_unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet)
        framer_config_path = os.path.join(script_directory, "configs", "svd_config.json")
        with open(framer_config_path) as f:
            framer_config = json.load(f)
        with init_empty_weights():
            framer_unet = UNetSpatioTemporalConditionModel.from_config(
                framer_config,
                torch_dtype=torch.float16,
                custom_resume=True,
            )
        
        if attention_mode == "xformers":
            from .models_diffusers.attention_processor import XFormersAttnProcessor
            framer_unet.set_attn_processor(XFormersAttnProcessor())

        framer_unet_sd = load_torch_file(framer_unet_path, device=load_device, safe_load=True)
        for name, param in framer_unet.named_parameters():
            set_module_tensor_to_device(framer_unet, name, device=load_device, dtype=base_dtype, value=framer_unet_sd[name])
        del framer_unet_sd

        #controlnet
        controlnet_path = folder_paths.get_full_path_or_raise("diffusion_models", controlnet)
        controlet_config_path = os.path.join(script_directory, "configs", "controlnet_config.json")
        with open(controlet_config_path) as f:
            controlnet_config = json.load(f)
        with init_empty_weights():
            controlnet = ControlNetSVDModel.from_config(controlnet_config)
        
        controlnet_sd = load_torch_file(controlnet_path, device=load_device, safe_load=True)
        for name, param in controlnet.named_parameters():
            set_module_tensor_to_device(controlnet, name, device=load_device, dtype=base_dtype, value=controlnet_sd[name])

        svd_xt_path = os.path.join(folder_paths.models_dir, "diffusers", "stable-video-diffusion-img2vid-xt-1-1")
        if not os.path.exists(svd_xt_path):
            log.info(f"Downloading SVD model to: {svd_xt_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1", 
                                allow_patterns=[f"*.json", "*fp16*"],
                                ignore_patterns=["*unet*"],
                                local_dir=svd_xt_path, 
                                local_dir_use_symlinks=False)

        #pipeline
        pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
            svd_xt_path,
            unet=framer_unet,
            controlnet=controlnet,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        )
        pipe.to(device)            

        compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            if compile_args["compile_unet"]:
                pipe.unet = torch.compile(pipe.unet, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_controlnet"]:
                pipe.controlnet = torch.compile(pipe.controlnet, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        return (pipe,)


class FramerTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_unet": ("BOOLEAN", {"default": True, "tooltip": "Compile the SVD unet"}),
                "compile_controlnet": ("BOOLEAN", {"default": False, "tooltip": "Compile the Framer controlnet"}),
            },
        }
    RETURN_TYPES = ("COMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "FramerWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, compile_unet, compile_controlnet, dynamo_cache_size_limit):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_unet": compile_unet,
            "compile_controlnet": compile_controlnet
        }

        return (compile_args, )

#region sampler

class FramerSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FRAMERMODEL",),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
                "num_frames": ("INT", {"default": 14, "min": 1, "max": 1024, "step": 4}),
                "steps": ("INT", {"default": 20, "min": 1}),
                "min_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "max_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_bucket_id": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "fps": ("INT", {"default": 7, "min": 1, "max": 60}),
                "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "controlnet_cond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "tracks": ("PREDTRACKS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "FramerWrapper"

    def process(self, model, start_image, end_image, controlnet_cond_scale, motion_bucket_id, fps, noise_aug_strength, steps, min_guidance_scale, max_guidance_scale, seed, num_frames, force_offload, tracks=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        unet = model.unet

        B, H, W, C = start_image.shape
        start_image = start_image.permute(0, 3, 1, 2).to(device)
        end_image = end_image.permute(0, 3, 1, 2).to(device)

        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        anchor_points_flag = tracks.get("anchor_points_flag") if tracks is not None else None
        

        mm.soft_empty_cache()
        gc.collect()

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #for name, param in transformer.named_parameters():
        #    print(name, param.data.device)
        unet.to(device)
        video_frames = model(
            start_image,
            end_image,
            # trajectory control
            with_control=True if tracks is not None else False,
            point_tracks=tracks["pred_tracks"] if tracks is not None else None,
            point_embedding=None,
            with_id_feature=False,
            controlnet_cond_scale=controlnet_cond_scale,
            # others
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            num_frames=num_frames,
            width=W,
            height=H,
            decode_chunk_size=2,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=steps,
            # track
            sift_track_update=False,
            anchor_points_flag=anchor_points_flag,
            output_type="pt",
        ).frames[0]

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        if force_offload:
            unet.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()
        print(video_frames.shape)
        out = video_frames.permute(0, 2, 3, 1).cpu().float()

        return out,

        # return ({
        #     "samples": out_latents
        #     },)

class FramerSift:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "start_image": ("IMAGE",),
                    "end_image": ("IMAGE",),
                    "num_frames": ("INT", {"default": 14, "min": 1, "max": 1024, "step": 1}),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "topk": ("INT", {"default": 5, "min": 1, "max": 100}),
                    "method": (["max_dist", "random", "max_score", "max_score_even"], {"default": "max_dist"}),
                    },
                }

    RETURN_TYPES = ("PREDTRACKS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pred_tracks", "visualization", "vis_frames")
    FUNCTION = "sift"
    CATEGORY = "FramerWrapper"

    def sift(self, start_image, end_image, num_frames, threshold, topk, method):
        device = mm.get_torch_device()
        from .models_diffusers.sift_match import interpolate_trajectory as sift_interpolate_trajectory
        from .models_diffusers.sift_match import sift_match
        
        B, H, W, C = start_image.shape
        #start_image = start_image.permute(0, 3, 1, 2).to(device)
        #end_image = end_image.permute(0, 3, 1, 2).to(device)

        # (f, topk, 2), f=2 (before interpolation)
        pred_tracks, vis_image = sift_match(
            start_image,
            end_image,
            thr=threshold,
            topk=topk,
            method=method,
        )

        # interpolate the tracks, following draganything gradio demo
        pred_tracks = sift_interpolate_trajectory(pred_tracks, num_frames=num_frames)

        anchor_points_flag = torch.zeros((num_frames, pred_tracks.shape[1])).to(pred_tracks.device)
        anchor_points_flag[0] = 1
        anchor_points_flag[-1] = 1

        vis_image_tensor = torch.from_numpy(vis_image).contiguous().unsqueeze(0).to(device).float() / 255.0
        #pred_tracks = pred_tracks.permute(1, 0, 2)  # (num_points, num_frames, 2)
        log.info(f"pred_tracks: {pred_tracks.shape}")

        vis_frames = get_vis_image(target_size=(H, W), points=pred_tracks.permute(1, 0, 2), num_frames=num_frames, side=20)
        
        vis_tensors = []
        for img in vis_frames:
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            vis_tensors.append(img)
        vis_frames_out = torch.stack(vis_tensors)
        vis_frames_out = vis_frames_out.permute(0, 2, 3, 1).cpu().float()

        pred_tracks = {
            "pred_tracks": pred_tracks,
            "anchor_points_flag": anchor_points_flag,
        }


        return (pred_tracks, vis_image_tensor, vis_frames_out,)
    
class CoordsToFramerTracking:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "coordinates": ("STRING", {"forceInput": True}),
                    "coordinates_2": ("STRING", {"forceInput": True}),
                    "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                    },
                }

    RETURN_TYPES = ("PREDTRACKS", "IMAGE",)
    RETURN_NAMES = ("pred_tracks", "vis_frames")
    FUNCTION = "convert"
    CATEGORY = "FramerWrapper"

    def convert(self, coordinates, coordinates_2, width, height):
        coords_list = []
        coords = json.loads(coordinates.replace("'", '"'))
        coords = [(coord['x'], coord['y']) for coord in coords]
        num_frames = len(coords)

        coords_2 = json.loads(coordinates_2.replace("'", '"'))
        coords_2 = [(coord['x'], coord['y']) for coord in coords_2]

        coords_list.append(coords)
        coords_list.append(coords_2)

        coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        coords_tensor= coords_tensor.permute(1, 0, 2)  # (num_frames, num_points, 2)
        #print("pred_tracks: ", coords_tensor.shape)
        vis_frames = get_vis_image(target_size=(width, height), points=coords_tensor.permute(1, 0, 2), num_frames=num_frames, side=20)
        
        vis_tensors = []
        for img in vis_frames:
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            vis_tensors.append(img) 
        vis_frames_out = torch.stack(vis_tensors)
        vis_frames_out = vis_frames_out.permute(0, 2, 3, 1).cpu().float()

        pred_tracks = {
            "pred_tracks": coords_tensor,
            "anchor_points_flag": None,
        }

        return (pred_tracks, vis_frames_out,)
    
#region VideoDecode
class HyVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "FramerWrapper"

    def decode(self, vae, samples, enable_vae_tiling, temporal_tiling_sample_size, spatial_tile_sample_min_size, auto_tile_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        latents = samples["samples"]
        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32


        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        if enable_vae_tiling:
            vae.enable_tiling()
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]
        else:
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]

        if expand_temporal_dim or video.shape[2] == 1:
            video = video.squeeze(2)

        vae.to(offload_device)
        mm.soft_empty_cache()

        if len(video.shape) == 5:
            video_processor = VideoProcessor(vae_scale_factor=8)
            video_processor.config.do_resize = False

            video = video_processor.postprocess_video(video=video, output_type="pt")
            out = video[0].permute(0, 2, 3, 1).cpu().float()
        else:
            out = (video / 2 + 0.5).clamp(0, 1)
            out = out.permute(0, 2, 3, 1).cpu().float()

        return (out,)

#region VideoEncode
class HyVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "FramerWrapper"

    def encode(self, vae, image, enable_vae_tiling, temporal_tiling_sample_size, auto_tile_size, spatial_tile_sample_min_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32

        image = (image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if enable_vae_tiling:
            vae.enable_tiling()
        latents = vae.encode(image).latent_dist.sample(generator)
        latents = latents * vae.config.scaling_factor
        vae.to(offload_device)
        print("encoded latents shape",latents.shape)


        return ({"samples": latents},)


NODE_CLASS_MAPPINGS = {
    "FramerModelLoader": FramerModelLoader,
    "FramerSampler": FramerSampler,
    "FramerTorchCompileSettings": FramerTorchCompileSettings,
    "FramerSift": FramerSift,
    "CoordsToFramerTracking": CoordsToFramerTracking,
    
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "FramerModelLoader": "Framer Model Loader",
    "FramerSampler": "Framer Sampler",
    "FramerTorchCompileSettings": "Framer Torch Compile Settings",
    "FramerSift": "Framer Sift",
    "CoordsToFramerTracking": "Coords To Framer Tracking",
    }