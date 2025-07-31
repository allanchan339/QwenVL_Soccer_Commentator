import outetts
import torch

interface = outetts.Interface(
    config=outetts.ModelConfig(
        model_path="OuteAI/Llama-OuteTTS-1.0-1B",
        tokenizer_path="OuteAI/Llama-OuteTTS-1.0-1B",
        interface_version=outetts.InterfaceVersion.V3,
        backend=outetts.Backend.HF,
        additional_model_config={
            "attn_implementation": "flash_attention_2"  # Enable flash attention if compatible
        },
        device="cuda",
        dtype=torch.bfloat16
    )
)

# Load the default **English** speaker profile
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker (Use this once)
# speaker = interface.create_speaker("path/to/audio.wav")
# interface.save_speaker(speaker, "speaker.json")

# Load your speaker from saved file
# speaker = interface.load_speaker("speaker.json")

# Generate speech & save to file
output = interface.generate(
    outetts.GenerationConfig(
        text="Hello, how are you doing?",
        speaker=speaker,
    )
)
output.save("output.wav") 