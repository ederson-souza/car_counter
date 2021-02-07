# Vehicle Counter with Detectron2

## Requirements
- Linux with Python ≥ 3.6
- CUDA
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- OpenCV
- Detectron2
  - For Detectron2 ensure you have gcc & g++ installed first.
- youtube_dl
- Pafy

## Installation and Usage

```
1. git clone https://github.com/edinhoadm/car_counter.git
2. cd car_counter
3. pip install -r requirements.txt

# If you don't have gcc and g++ do the following, else skip to step n. 7:
4. sudo apt update
5. sudo apt install gcc
6. sudo apt install g++

7. pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
8. python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

After that, just run `python car_detection.py`
The processed video will be stored in the folder named "output".



