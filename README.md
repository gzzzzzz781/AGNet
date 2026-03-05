# AGNet
Official implement of paper "AGNet: Attention Guided Network for Single HDR Reconstruction"

## Usage

### Requirements
* Python 3.9
* PyTorch 
* Torchvision 

Install the require dependencies:
```bash
conda create -n hdr_transformer_pytorch python=3.9
conda activate hdr_transformer_pytorch
pip install -r requirements.txt
```
### Dataset
1. Download the dataset (include the NTIRE2021 dataset and Cityscape dataset) from https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/

2. Prepare the corpped training set by running:
```
cd ./dataset
python gen_crop_data.py
```

### Training & Test & Evaluaton

To train the model, run:
```
python train.py
```
To test, run:
```
python fullimagetest.py 
```
To evaluate, run:
```
python evaluate.py 
```
To visualize, run:
```
python visualize.py 
```

## Acknowledgement
Our work is inspired the following works and uses parts of their official implementations:

* [Restormer](https://github.com/swz30/Restormer)

We thank the respective authors for open sourcing their methods.

## Contact
If you have any questions, feel free to contact Zhou Gong at zhougong@mail.nwpu.edu.cn.
