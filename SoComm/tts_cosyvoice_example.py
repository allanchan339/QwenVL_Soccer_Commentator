import sys
import torchaudio
import os

# Add third-party and project directories to sys.path for imports
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def run_zero_shot_example(cosyvoice, text, prompt_text, prompt_audio, prefix, output_dir):
    """Run zero-shot inference and save results with a given prefix inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio, stream=False)):
        output_path = os.path.join(output_dir, f'{prefix}_{i}.wav')
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f'Saved: {output_path}')

def run_instruct_example(cosyvoice, text, instruction, prompt_audio, prefix, output_dir):
    """Run instruct inference and save results with a given prefix inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(cosyvoice.inference_instruct2(text, instruction, prompt_audio, stream=False)):
        output_path = os.path.join(output_dir, f'{prefix}_{i}.wav')
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f'Saved: {output_path}')

def main():
    # Output directory for all audio files
    output_dir = 'temp_audio/cosyvoice'

    # Initialize the TTS model
    model_path = 'pretrained_models/CosyVoice2-0.5B'
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # Example 1: Zero-shot with default prompt
    prompt_path = 'CosyVoice/asset/zero_shot_prompt.wav'
    prompt_speech_16k = load_wav(prompt_path, 16000)
    text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    prompt_text = '希望你以后能够做的比我还好呦。'  # from asset/zero_shot_prompt.wav
    run_zero_shot_example(cosyvoice, text, prompt_text, prompt_speech_16k, 'zero_shot', output_dir)

    # Example 2: Zero-shot with a different audio prompt
    print("\n--- CosyVoice2 Zero-Shot Example: ballisrounded.wav prompt ---")
    new_prompt_path = 'assets/audio_prompt/ballisrounded.wav'
    new_prompt_speech_16k = load_wav(new_prompt_path, 16000)
    new_text = '喂！三点几嚟，饮茶先呢，做咁多都冇用嘅，老细唔锡你嘅呢'  # "The ball is round, the match is full of unpredictable exciting moments."
    new_prompt_text = '老套既四個字都忍吾住講埋出黎，唉，波係圓嘅，冇法啦'  # (or a short description of the prompt audio)
    run_zero_shot_example(cosyvoice, new_text, new_prompt_text, new_prompt_speech_16k, 'ballisrounded_zero_shot', output_dir)

    # Example 3: Instruct usage
    print("\n--- CosyVoice2 Instruct Example: 广东话 ---")
    run_instruct_example(cosyvoice, new_text, '用广东话说这句话', prompt_speech_16k, 'instruct', output_dir)

    # Example 4: Instruct usage new text and new prompt
    print("\n--- CosyVoice2 Instruct Example: 广东话 ---")
    run_instruct_example(cosyvoice, new_text, '用广东话说这句话', new_prompt_speech_16k, 'instruct_new', output_dir)

if __name__ == '__main__':
    main()
