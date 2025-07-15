import sys
import torchaudio

# Add third-party and project directories to sys.path for imports
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav


def main():
    # Initialize the TTS model
    model_path = 'pretrained_models/CosyVoice2-0.5B'
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # Load the prompt audio (16kHz)
    prompt_path = 'CosyVoice/asset/zero_shot_prompt.wav'
    prompt_speech_16k = load_wav(prompt_path, 16000)

    # Texts for TTS
    text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    prompt_text = '希望你以后能够做的比我还好呦。' # from asset/zero_shot_prompt.wav

    # Run zero-shot inference and save each result as a wav file
    for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False)):
        output_path = f'zero_shot_{i}.wav'
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f'Saved: {output_path}')

    # --- New Example: Using a different audio prompt ---
    print("\n--- CosyVoice2 Zero-Shot Example: ballisrounded.wav prompt ---")
    new_prompt_path = 'assets/audio_prompt/ballisrounded.wav'
    new_prompt_speech_16k = load_wav(new_prompt_path, 16000)
    new_text = '足球係圆的，比赛充满了不可预测的精彩瞬间。'  # "The ball is round, the match is full of unpredictable exciting moments."
    new_prompt_text = '老套既四個字都忍吾住講埋出黎，唉，波係圓嘅，冇法啦'  # (or a short description of the prompt audio)
    for i, result in enumerate(cosyvoice.inference_zero_shot(new_text, new_prompt_text, new_prompt_speech_16k, stream=False)):
        output_path = f'ballisrounded_zero_shot_{i}.wav'
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
