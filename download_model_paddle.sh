#!/bin/bash

# Create pretrained_models/paddlespeech directory if it doesn't exist
mkdir -p pretrained_models/paddlespeech
cd pretrained_models/paddlespeech

wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_canton_onnx_1.4.0.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip

unzip fastspeech2_canton_onnx_1.4.0.zip
unzip hifigan_csmsc_onnx_0.2.0.zip
