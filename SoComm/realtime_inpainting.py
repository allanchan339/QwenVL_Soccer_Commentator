# realtime_inpainting.py
"""
Handles real-time inpainting with avatar preprocessing and caching.
This is a wrapper around the realtime_inference.py functionality for web UI integration.
"""
import os
import cv2
import numpy as np
import torch
import imageio
import glob
import pickle
import copy
import json
import shutil
import threading
import queue
import time
from types import SimpleNamespace as Namespace
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from tqdm import tqdm


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """Extract frames from video to images."""
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    """Create multiple directories."""
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


class RealtimeAvatar:
    """Real-time avatar class for MuseTalk with preprocessing and caching."""
    
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, 
                 device=None, vae=None, unet=None, pe=None, weight_dtype=None, 
                 audio_processor=None, whisper=None, timesteps=None, version="v15",
                 extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation
        self.device = device
        self.vae = vae
        self.unet = unet
        self.pe = pe
        self.weight_dtype = weight_dtype
        self.audio_processor = audio_processor
        self.whisper = whisper
        self.timesteps = timesteps
        self.version = version
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        
        # Set base path based on version
        if version == "v15":
            self.base_path = f"./results/{version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"
        
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": version,
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width
        }
        
        self.idx = 0
        self.init()

    def init(self):
        """Initialize avatar - either load existing or create new."""
        if self.preparation:
            if os.path.exists(self.avatar_path):
                # Avatar exists, load it
                self.load_existing_avatar()
            else:
                # Create new avatar
                print(f"Creating avatar: {self.avatar_id}")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                raise FileNotFoundError(f"{self.avatar_id} does not exist, set preparation=True to create it")
            
            # Check if parameters match
            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)
            
            if (avatar_info['bbox_shift'] != self.avatar_info['bbox_shift'] or
                avatar_info['version'] != self.avatar_info['version']):
                print(f"Parameters changed for {self.avatar_id}, recreating...")
                shutil.rmtree(self.avatar_path)
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
            else:
                self.load_existing_avatar()

    def load_existing_avatar(self):
        """Load existing avatar data from disk."""
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        """Prepare avatar materials (frames, latents, masks)."""
        print("Preparing avatar materials...")
        
        # Save avatar info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        # Extract frames from video
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            # Copy existing images
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        # Extract landmarks and create latents
        print("Extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        # Create cyclic lists for smooth looping
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Initialize face parser
        fp = FaceParsing(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width
        )
        
        # Prepare masks for blending
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.version == "v15":
                mode = self.parsing_mode
            else:
                mode = "raw"
            
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        # Save all materials
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)

    def process_frames(self, res_frame_queue, video_len, skip_save_images, output_dir):
        """Process generated frames in a separate thread."""
        print(f"Processing {video_len} frames...")
        os.makedirs(output_dir, exist_ok=True)
        
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                cv2.imwrite(f"{output_dir}/{str(self.idx).zfill(8)}.png", combine_frame)
            
            self.idx = self.idx + 1

    def inference(self, audio_path, out_vid_name, fps, skip_save_images=True, output_dir=None):
        """Run real-time inference with the avatar."""
        if output_dir is None:
            output_dir = f"{self.avatar_path}/tmp"
        
        os.makedirs(output_dir, exist_ok=True)
        print("Starting real-time inference...")
        
        # Extract audio features
        start_time = time.time()
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        print(f"Audio processing took {(time.time() - start_time) * 1000:.2f}ms")
        
        # Setup inference
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        
        # Start processing thread
        process_thread = threading.Thread(
            target=self.process_frames, 
            args=(res_frame_queue, video_num, skip_save_images, output_dir)
        )
        process_thread.start()

        # Run inference
        gen = datagen(
            whisper_chunks,
            self.input_latent_list_cycle,
            self.batch_size
        )
        
        start_time = time.time()
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        
        # Wait for processing to complete
        process_thread.join()

        processing_time = time.time() - start_time
        print(f'Total processing time for {video_num} frames: {processing_time:.2f}s')
        print(f'Average time per frame: {processing_time/video_num*1000:.2f}ms')

        # Generate final video if requested
        if out_vid_name is not None and not skip_save_images:
            output_video = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
            
            # Create video from frames
            images = []
            files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')], 
                          key=lambda x: int(x.split('.')[0]))
            
            for file in files:
                filename = os.path.join(output_dir, file)
                images.append(imageio.imread(filename))
            
            # Write video
            imageio.mimwrite('temp.mp4', images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
            
            # Add audio
            from moviepy.editor import VideoFileClip, AudioFileClip
            video_clip = VideoFileClip('temp.mp4')
            audio_clip = AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(output_video, codec='libx264', audio_codec='aac', fps=fps)
            
            # Cleanup
            os.remove('temp.mp4')
            shutil.rmtree(output_dir)
            print(f"Result saved to {output_video}")
            return output_video
        
        return None


def create_realtime_avatar(
    avatar_id, video_path, bbox_shift=0, batch_size=20, preparation=True,
    device=None, vae=None, unet=None, pe=None, weight_dtype=None, 
    audio_processor=None, whisper=None, timesteps=None, version="v15",
    extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90
):
    """Create a real-time avatar instance."""
    return RealtimeAvatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=batch_size,
        preparation=preparation,
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        weight_dtype=weight_dtype,
        audio_processor=audio_processor,
        whisper=whisper,
        timesteps=timesteps,
        version=version,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )


def realtime_inference(
    avatar_id, audio_path, video_path, bbox_shift=0, batch_size=20, preparation=True,
    device=None, vae=None, unet=None, pe=None, weight_dtype=None, 
    audio_processor=None, whisper=None, timesteps=None, version="v15",
    extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90,
    fps=25, skip_save_images=True, output_vid_name=None
):
    """High-level function for real-time inference."""
    # Create avatar
    avatar = create_realtime_avatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=batch_size,
        preparation=preparation,
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        weight_dtype=weight_dtype,
        audio_processor=audio_processor,
        whisper=whisper,
        timesteps=timesteps,
        version=version,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )
    
    # Run inference
    result = avatar.inference(
        audio_path=audio_path,
        out_vid_name=output_vid_name,
        fps=fps,
        skip_save_images=skip_save_images
    )
    
    return result 