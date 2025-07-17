# Requirement 
1. NVidia GPU with CUDA support (Here we use 8*RTX4090)
2. Ubuntu 20.04 or higher
3. Driver version >= 570.133 
4. CUDA version >= 12.0
5. The environment must be created with Python 3.10 (CosyVoice-ttsfrd requires Python 3.10)

# Installation 
## Git 
1. Clone the repository:
```bash
git clone https://github.com/XX.git --depth 1  
cd XX
git submodule update --init --recursive
```

## Conda
1. Install Miniconda or Anaconda.
`conda env create -f environemt.yml`

2. Activate the environment:
```bash
conda activate SoCommVoice
```

## Additional Dependencies
### Install additional dependencies for musetalk:
```bash
# Install dependencies related to musetalk
pip install --no-cache-dir -U openmim
mim install mmengine 
mim install "mmcv==2.1.0"  # ref to https://mmcv.readthedocs.io/en/latest/get_started/installation.html for more details, mmcv 2.2.0 is the only stable version of cuda 12.1 (or above?) and torch 2.4 (or above)? buggy as they assert versioning  
mim install "mmdet==3.2.0" # mmdet 3.3.0 requires mmcv<2.2.0,>=2.0.0rc4; extra == "mim", but you have mmcv 2.2.0 which is incompatible. (but mmcv 2.2.0 is the only stable version for cuda 12+ & torch 2.4+)
mim install "mmpose>=1.1.0" # mmpose 1.3.2 requires mmdet<3.3.0,>=3.0.0; extra == "mim", but you have mmdet 3.3.0 which is incompatible. (same reason as above)

#optional
pip install -U "huggingface_hub[cli]" # u also need to login to download some models
```

### Install additional dependencies for CosyVoice:
```bash
# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

### Install additional dependencies for PaddleSpeech:
```bash
pip install paddlespeech paddlepaddle --no-deps
pip install yacs g2p-en opencc pypinyin pypinyin-dict opencc-python-reimplemented braceexpand ToJyutping webrtcvad zhon timer
```

## Download pre-trained models
### Download the pre-trained models and install CosyVoice-ttsfrd:
```bash
# Download the CosyVoice model
python download_model_cosyvoice.py

# Install the CosyVoice-ttsfrd model (Optional, if not installed, wetext will be used)
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Download the pre-trained models and install MuseTalk:
```bash
# Download the MuseTalk model
cd MuseTalk
sh ./download_weights.sh
```