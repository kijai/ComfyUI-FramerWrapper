# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import VaeImageProcessor
# from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .models_diffusers.controlnet_svd import ControlNetSVDModel
# from cotracker.predictor import CoTrackerPredictor, sample_trajectories, generate_gassian_heatmap
from .models_diffusers.utils import generate_gassian_heatmap
from .models_diffusers.sift_match import point_tracking, interpolate_trajectory

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from comfy.utils import ProgressBar


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class StableVideoDiffusionInterpControlPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionInterpControlPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        controlnet: Optional[ControlNetSVDModel] = None,
        pose_encoder: Optional[torch.nn.Module] = None,
        offload_device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
            pose_encoder=pose_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.offload_device = offload_device

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

        # We normalize the image before resizing to match with the original implementation.
        # Then we unnormalize it after resizing.
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        self.vae.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        self.vae.to(self.offload_device)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image_end: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        # for points
        with_control: bool = True,
        point_tracks: Optional[torch.FloatTensor] = None,
        point_embedding: Optional[torch.FloatTensor] = None,
        with_id_feature: bool = False,  # NOTE: whether to use the id feature for controlnet
        controlnet_cond_scale: float = 1.0,
        controlnet_step_range: List[float] = [0, 1],
        # others
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        middle_max_guidance: bool = False,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        # update track
        sift_track_update: bool = False,
        sift_track_update_with_time: bool = True,
        sift_track_feat_idx: List[int] = [2, ],
        sift_track_dist: int = 5,
        sift_track_double_check_thr: float = 2,
        anchor_points_flag: Optional[torch.FloatTensor] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionInterpControlPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionInterpControlPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)
        self.check_inputs(image_end, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        self.image_encoder.to(device)
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_end_embeddings = self._encode_image(image_end, device, num_videos_per_prompt, do_classifier_free_guidance)
        self.image_encoder.to(self.offload_device)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        #image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image * 2.0 - 1.0
        image = image + noise_aug_strength * noise
        
        # also for image_end
        #image_end = self.image_processor.preprocess(image_end, height=height, width=width)
        noise = randn_tensor(image_end.shape, generator=generator, device=image_end.device, dtype=image_end.dtype)
        image_end = image_end * 2.0 - 1.0
        image_end = image_end + noise_aug_strength * noise
        
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        if with_control:
            # create controlnet input
            video_gaussion_map = generate_gassian_heatmap(point_tracks, image_size=(width, height))
            controlnet_image = video_gaussion_map.unsqueeze(0)  # (1, f, c, h, w)
            controlnet_image = controlnet_image.to(device, dtype=image_embeddings.dtype)
            controlnet_image = torch.cat([controlnet_image] * 2, dim=0)

            point_embedding = point_embedding.to(device).to(image_embeddings.dtype) if point_embedding is not None else None
            point_tracks = point_tracks.to(device).to(image_embeddings.dtype)  # (f, p, 2)

            assert point_tracks.shape[0] == num_frames, f"point_tracks.shape[0] != num_frames, {point_tracks.shape[0]} != {num_frames}"
            # if point_tracks.shape[0] != num_frames:
            #     # interpolate the point_tracks to the number of frames
            #     point_tracks = rearrange(point_tracks[None], 'b f p c -> b p f c')
            #     point_tracks = torch.nn.functional.interpolate(point_tracks, size=(num_frames, point_tracks.shape[-1]), mode='bilinear', align_corners=False)[0]
            #     point_tracks = rearrange(point_tracks, 'p f c -> f p c')

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)
        # also for image_end
        image_end_latents = self._encode_vae_image(image_end, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_end_latents = image_end_latents.to(image_end_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        # image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # Concatenate the `conditional_latents` with the `noisy_latents`.
        # conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
        image_latents = image_latents.unsqueeze(1)  # (1, 1, 4, h, w)
        bsz, num_frames, _, latent_h, latent_w = latents.shape
        bsz_cfg = bsz * 2
        mask_token = self.unet.mask_token
        conditional_latents_mask = mask_token.repeat(bsz_cfg, num_frames-2, 1, latent_h, latent_w)
        image_end_latents = image_end_latents.unsqueeze(1)
        image_latents = torch.cat([image_latents, conditional_latents_mask, image_end_latents], dim=1)

        # Concatenate additional mask channel
        mask_channel = torch.ones_like(image_latents[:, :, 0:1, :, :])
        mask_channel[:, 0:1, :, :, :] = 0
        mask_channel[:, -1:, :, :, :] = 0
        image_latents = torch.cat([image_latents, mask_channel], dim=2)

        # concate the conditions
        image_embeddings = torch.cat([image_embeddings, image_end_embeddings], dim=1)

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)  # (1, 14)
        if middle_max_guidance:
            # big in middle, small at the beginning and end
            guidance_scale = torch.cat([guidance_scale, guidance_scale.flip(1)], dim=1)
            # interpolate the guidance scale, from [1, 2*frames] to [1, frames] 
            guidance_scale = torch.nn.functional.interpolate(guidance_scale.unsqueeze(0), size=num_frames, mode='linear', align_corners=False)[0]


        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if with_control and sift_track_update:
            num_tracks = point_tracks.shape[1]
            anchor_point_dict = {}
            for frame_idx in range(num_frames):
                anchor_point_dict[frame_idx] = {}
                for point_idx in range(num_tracks):
                    # add the start and end point
                    if frame_idx in [0, num_frames - 1]:
                        anchor_point_dict[frame_idx][point_idx] = point_tracks[frame_idx][point_idx]
                    else:
                        anchor_point_dict[frame_idx][point_idx] = None

        with_control_global = with_control
        comfy_pbar = ProgressBar(num_inference_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # NOTE: set the range for control
                if with_control_global:
                    if controlnet_step_range[0] <= i / num_inference_steps < controlnet_step_range[1]:
                        with_control = True
                    else:
                        with_control = False
                    # print(f"step={i / num_inference_steps}, with_control={with_control}")

                if with_control and sift_track_update and i > 0:
                    # update the point tracks
                    track_list = []
                    for point_idx in range(num_tracks):
                        # get the anchor points
                        current_track = []
                        current_time_to_interp = []
                        for frame_idx in range(num_frames):
                            if anchor_points_flag[frame_idx][point_idx] == 1:
                                current_track.append(anchor_point_dict[frame_idx][point_idx].cpu())
                                if sift_track_update_with_time:
                                    current_time_to_interp.append(frame_idx / (num_frames - 1))

                        current_track = torch.stack(current_track, dim=0).unsqueeze(1)  # (f, 1, 2)
                        # interpolate the anchor points to obtain trajectory
                        current_time_to_interp = np.array(current_time_to_interp) if sift_track_update_with_time else None
                        current_track = interpolate_trajectory(current_track, num_frames=num_frames, t=current_time_to_interp)
                        track_list.append(current_track)
                    point_tracks = torch.concat(track_list, dim=1).to(device).to(image_embeddings.dtype)  # (f, p, 2)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                down_block_res_samples = mid_block_res_sample = None
                if with_control:
                    if i == 0:
                        print(f"controlnet_cond_scale: {controlnet_cond_scale}")
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        controlnet_cond=controlnet_image,
                        added_time_ids=added_time_ids,
                        conditioning_scale=controlnet_cond_scale,
                        point_embedding=point_embedding if with_id_feature else None,  # NOTE 
                        point_tracks=point_tracks,
                        guess_mode=False,
                        return_dict=False,
                    )
                else:
                    if i == 0:
                        print("Controlnet is not used")

                kwargs = {}

                outputs = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                    **kwargs,
                )

                noise_pred, intermediate_features = outputs

                if with_control and sift_track_update:
                    # shape: [b*f, c, h, w], b=2 for cfg
                    matching_features = []
                    for feat_idx in sift_track_feat_idx:
                        feat = intermediate_features[feat_idx]
                        feat = F.interpolate(feat, (height, width), mode='bilinear')
                        matching_features.append(feat)

                    matching_features = torch.cat(matching_features, dim=1)  # [b*f, c, h, w]

                    # shape: [b*f, c, h, w]
                    # self.guidance_scale: [1, f, 1, 1, 1]
                    # matching_features: 
                    assert do_classifier_free_guidance
                    matching_features = rearrange(matching_features, '(b f) c h w -> b f c h w', b=2)

                    # # strategy 1: discard the unconditional branch feature maps
                    # matching_features = matching_features[1].unsqueeze(dim=0)  # (b, f, c, h, w), b=1
                    # # strategy 2: concat pos and neg branch feature maps for motion-sup and point tracking
                    # matching_features = torch.cat([matching_features[0], matching_features[1]], dim=1).unsqueeze(dim=0)  # (b, f, 2c, h, w), b=1
                    # # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
                    # coef = self.guidance_scale / (2 * self.guidance_scale - 1.0)
                    # coef = coef.squeeze(dim=0)
                    # matching_features = torch.cat(
                    #     [(1 - coef) * matching_features[0], coef * matching_features[1]], dim=1,
                    # ).unsqueeze(dim=0)  # (b, f, 2c, h, w), b=1
                    # strategy 4: same as cfg
                    matching_features = matching_features[0] + self.guidance_scale.squeeze(0) * (matching_features[1] - matching_features[0])
                    matching_features = matching_features.unsqueeze(dim=0)  # (b, f, c, h, w), b=1

                    # perform point matching in intermediate frames
                    feature_start = matching_features[:, 0]
                    feature_end = matching_features[:, -1]
                    hanlde_points_start = point_tracks[0]  # (f, p, 2) -> (p, 2)
                    hanlde_points_end = point_tracks[-1]  # (f, p, 2) -> (p, 2)
                    for frame_idx in range(1, num_frames - 1):
                        feature_frame = matching_features[:, frame_idx]
                        handle_points = point_tracks[frame_idx]  # (f, p, 2) -> (p, 2)
                        # forward matching
                        handle_points_forward = point_tracking(feature_start, feature_frame, handle_points, hanlde_points_start, sift_track_dist)
                        # backward matching
                        handle_points_backward = point_tracking(feature_end, feature_frame, handle_points, hanlde_points_end, sift_track_dist)

                        # bi-directional check
                        for point_idx, (point_forward, point_backward) in enumerate(zip(handle_points_forward, handle_points_backward)):
                            if torch.norm(point_forward - point_backward) < sift_track_double_check_thr:
                                # update the point
                                # point_tracks[frame_idx][point_idx] = (point_forward + point_backward) / 2
                                anchor_point_dict[frame_idx][point_idx] = (point_forward + point_backward) / 2
                                anchor_points_flag[frame_idx][point_idx] = 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    comfy_pbar.update(1)

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
                # self.vae.to(dtype=torch.float32)
                # latents = latents.to(torch.float32)
            self.vae.to(device)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            self.vae.to(self.offload_device)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionInterpControlPipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
