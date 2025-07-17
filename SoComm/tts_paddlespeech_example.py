import os
import shutil
from paddlespeech.cli.tts import TTSExecutor

# Output file name
output_filename = 'api_offline.wav'

# Run TTS and generate audio in the current directory
tts_executor = TTSExecutor()
wav_file = tts_executor(
    text='喂！三点几嚟，饮茶先呢，做咁多都冇用嘅，老细唔锡你嘅呢',
    output=output_filename,
    am='fastspeech2_canton',
    am_ckpt='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/fastspeech2_canton.onnx',
    phones_dict='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/phone_id_map.txt',
    speaker_dict='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/speaker_id_map.txt',
    voc='hifigan_csmsc',
    voc_ckpt='pretrained_models/paddlespeech/hifigan_csmsc_onnx_0.2.0/hifigan_csmsc.onnx',
    lang='canton',
    spk_id=10,
    use_onnx=True,
    cpu_threads=16)

# Ensure output directory exists
output_dir = os.path.join('temp_audio', 'paddlespeech')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)

# Move the generated file to the target directory
if os.path.exists(output_filename):
    shutil.move(output_filename, output_path)
    print(f"Moved output to {output_path}")
else:
    print(f"Output file {output_filename} not found.")