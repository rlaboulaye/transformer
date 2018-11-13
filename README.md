# transformer

Developped with python3.6 and CUDA 9.2

The requirements can be installed with the following command:
pip install -r requirements.txt

Torch needs to be compatible with CUDA 9.2 and so should be install as follows:
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

The repo is also dependent on a pretrained spacy model that can be installed as follows:
python -m spacy download en
