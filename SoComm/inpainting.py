# inpainting.py
"""
Handles inpainting, face parsing, and video processing logic.
"""
import os
import cv2
import numpy as np
import torch
import imageio
import glob
import pickle
import copy
from types import SimpleNamespace as Namespace
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm

def debug_inpainting(
    video_path, bbox_shift, extra_margin=10, parsing_mode="jaw",
    left_cheek_width=90, right_cheek_width=90,
    device=None, vae=None, unet=None, pe=None, weight_dtype=None, timesteps=None,
    result_dir='./results/debug', fps=25, batch_size=1, output_vid_name='', use_saved_coord=False,
    audio_padding_length_left=2, audio_padding_length_right=2, version="v15"
):
    args = Namespace(
        result_dir=result_dir,
        fps=fps,
        batch_size=batch_size,
        output_vid_name=output_vid_name,
        use_saved_coord=use_saved_coord,
        audio_padding_length_left=audio_padding_length_left,
        audio_padding_length_right=audio_padding_length_right,
        version=version,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )
    os.makedirs(args.result_dir, exist_ok=True)
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    x1, y1, x2, y2 = bbox
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

def inference(
    audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw",
    left_cheek_width=90, right_cheek_width=90,
    device=None, vae=None, unet=None, pe=None, weight_dtype=None,
    audio_processor=None, whisper=None, timesteps=None, progress=None,
    result_dir='./results/output', fps=25, batch_size=8, output_vid_name='', use_saved_coord=False,
    audio_padding_length_left=2, audio_padding_length_right=2, version="v15"
):
    args = Namespace(
        result_dir=result_dir,
        fps=fps,
        batch_size=batch_size,
        output_vid_name=output_vid_name,
        use_saved_coord=use_saved_coord,
        audio_padding_length_left=audio_padding_length_left,
        audio_padding_length_right=audio_padding_length_right,
        version=version,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)
    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        reader = imageio.get_reader(video_path)
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else:
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
    fps = 25
    output_video = 'temp.mp4'
    def is_valid_image(file):
        import re
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)
    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))
    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
    input_video = './temp.mp4'
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']
    reader.close()
    frames = images
    video_clip = VideoFileClip(input_video)
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)
    os.remove("temp.mp4")
    print(f"result is save to {output_vid_name}")
    return output_vid_name, bbox_shift_text 