import sys
import subprocess
import os
import argparse

# Add MuseTalk directory to sys.path
muse_talk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MuseTalk')
sys.path.append(muse_talk_dir)


def main():
    parser = argparse.ArgumentParser(description="MuseTalk inference runner (Python substitute for inference.sh)")
    parser.add_argument('--version', type=str, default='v1.5', choices=['v1.0', 'v1.5'], help="Model version (default: v1.5)")
    parser.add_argument('--mode', type=str, default='realtime', choices=['normal', 'realtime'], help="Inference mode (default: realtime)")
    args = parser.parse_args()

    version = args.version
    mode = args.mode

    # All paths are now relative to MuseTalk directory
    if mode == "normal":
        config_path = os.path.join(muse_talk_dir, "configs/inference/test.yaml")
        result_dir = os.path.join(muse_talk_dir, "results/test")
        script_name = "scripts.inference"
    else:
        config_path = os.path.join(muse_talk_dir, "configs/inference/realtime.yaml")
        result_dir = os.path.join(muse_talk_dir, "results/realtime")
        script_name = "scripts.realtime_inference"

    if version == "v1.0":
        model_dir = os.path.join(muse_talk_dir, "models/musetalk")
        unet_model_path = os.path.join(model_dir, "pytorch_model.bin")
        unet_config = os.path.join(model_dir, "musetalk.json")
        version_arg = "v1"
    elif version == "v1.5":
        model_dir = os.path.join(muse_talk_dir, "models/musetalkV15")
        unet_model_path = os.path.join(model_dir, "unet.pth")
        unet_config = os.path.join(model_dir, "musetalk.json")
        version_arg = "v15"
    else:
        print("Invalid version specified. Please use v1.0 or v1.5.")
        sys.exit(1)

    cmd_args = [
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--unet_model_path", unet_model_path,
        "--unet_config", unet_config,
        "--version", version_arg
    ]

    if mode == "realtime":
        cmd_args += ["--fps", "25", "--version", version_arg]

    # Set PYTHONPATH to include MuseTalk directory
    env = os.environ.copy()
    env["PYTHONPATH"] = muse_talk_dir + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    cmd = [sys.executable, "-m", script_name] + cmd_args
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True,env=env)

if __name__ == "__main__":
    main() 