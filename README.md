# Soccer Commentary & Talking Head Video Generation Demo

This repository demonstrates an AI-powered system that generates soccer commentary with synchronized talking head videos using advanced speech synthesis and lip-sync technology.

## Demo Videos

### Pure Lip Sync Demo
![Pure Lip Sync](assets/video/pure_lip_sync.mp4)

### Commentary Results
![Commentary Results](assets/video/commentary_results.mp4)

### Full Demo Recording
![Full Demo](assets/video/2025-08-27%2014-27-28.mov)

## Features
- **AI Soccer Commentary**: Generate realistic soccer match commentary using LLM
- **Talking Head Generation**: Create synchronized talking head videos with lip-sync
- **High-Quality Audio**: Advanced TTS (Text-to-Speech) synthesis
- **Real-time Processing**: Optimized for GPU acceleration

# Requirement 
1. NVidia GPU with CUDA support (1*RTX4060 is enough)
2. Ubuntu 20.04 or higher
3. Driver version >= 570.133 
4. CUDA version >= 12.0
5. The environment must be created with Python 3.10 (CosyVoice-ttsfrd requires Python 3.10)
6. [ModelScope API](https://www.modelscope.cn/my/myaccesstoken) key is required for LLM.

# Installation 
## Git 
1. Clone the repository:
```bash
git clone https://github.com/XX.git --depth 1  
cd XX
git submodule update --init --recursive
```

## Conda
1. Install Miniconda or Anaconda (environment.yml is for pytorch 2.1.2, environment_torch2.4.yml is for pytorch 2.4.1).
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
```

#### Install mmdet (For Pytorch 2.4.1 only)
Then we need to install mmdet and mmpose from source code and comment out the compatibility check in init.py. Otherwise, assertion error will be raised.

```bash
cd mmdetection
# Comment out the compatibility check in init.py
nano {python_path}/lib/python3.10/site-packages/mmdet/__init__.py 
```

Change the line 17 from:
```python        
and mmcv_version < digit_version(mmcv_maximum_version)), \
```
to:
```python        
and mmcv_version <= digit_version(mmcv_maximum_version)), \
```

#### Install mmpose
```bash
mim install "mmpose>=1.1.0" # not exist in conda-forge
```

<!-- ### Install additional dependencies for CosyVoice: (Ignored as yet implemented)
```bash
# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
``` -->

<!-- ### Install additional dependencies for PaddleSpeech (Ignored):
```bash
pip install paddlespeech paddlepaddle --no-deps
pip install yacs g2p-en opencc pypinyin pypinyin-dict opencc-python-reimplemented braceexpand ToJyutping webrtcvad zhon timer
``` -->

## Download pre-trained models
<!-- ### Download the pre-trained models and install CosyVoice-ttsfrd (Ignored as not required):
```bash
# Download the CosyVoice model
python download_model_cosyvoice.py

# Install the CosyVoice-ttsfrd model (Optional, if not installed, wetext will be used)
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```
 -->
### Download the pre-trained models and install MuseTalk:
```bash
# Download the MuseTalk model
sh ./download_weights.sh
```
