from paddlespeech.cli.tts import TTSExecutor

tts_executor = TTSExecutor()
wav_file = tts_executor(
    text='喂！三点几嚟，饮茶先嚟，做咁多都冇用嘅，老细唔锡你嘅嚟',
    output='api_offline.wav',
    am='fastspeech2_canton',
    am_ckpt='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/fastspeech2_canton.onnx',
    phones_dict='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/phone_id_map.txt',
    speaker_dict='pretrained_models/paddlespeech/fastspeech2_canton_onnx_1.4.0/speaker_id_map.txt',
    voc='hifigan_csmsc',
    voc_ckpt='pretrained_models/paddlespeech/hifigan_csmsc_onnx_0.2.0/hifigan_csmsc.onnx',
    lang='canton',
    spk_id=10,
    use_onnx=True,
    cpu_threads=2)