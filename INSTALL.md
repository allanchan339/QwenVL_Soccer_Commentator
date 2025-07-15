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
mim install "mmcv==2.1.0" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Install additional dependencies for CosyVoice:
```bash
pip install wget # although conda should install wget correctly, unfortunely it didnt.
# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

### Install additional dependencies for PaddleSpeech:
```bash
pip install paddlespeech paddlepaddle --no-deps
```

### Download the pre-trained models and install CosyVoice-ttsfrd:
```bash
# Download the CosyVoice model
python download_model_cosyvoice.py

cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```
