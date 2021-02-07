pip install -r requirements.txt
sudo apt update
sudo apt install gcc
sudo apt install g++
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
