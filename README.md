# HyBrIK-Helper

## Install

1. install hybrik and downlowd pretrained models on you mechine: [see HyBrIK official guide](https://github.com/Jeff-sjtu/HybrIK#installation-instructions)

2. clone this repository and create soft links
    ```
    git clone ...
    cd HyBrIK-helper
    ln -s /path/to/HyBrIK/hybrik hybrik
    ln -s /path/to/HyBrIK/model_files model_files
    ln -s /path/to/HyBrIK/pretrained_models pretrained_models 
    ``` 

3. dowload SMPL model: Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack mpips_smplify_public_v2.zip.

    Copy the smpl model.
    ```
    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models
    ```
    Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

## Estimate SMPL parameter from dataset

1. create a configuaration file and a script to load/save dataset (you can refer to the 3 examples we provide):
    Note that only zju_mocap performs stably under multi-view configuration, the other two dataset should be estimate in a mono-view manner. 

    1. ZJU-MoCap:
        - config: `configs/datasets/zju_mocap/387/cam1-view23.yaml`
        - script: `dataset_io/zju_mocap.py`

    2. GeneBody:
        - config: `configs/datasets/genebody/dilshod/cam0-mono.yaml`
        - script: `dataset_io/genebody.py`
    
    3. HuMMan:
        - config: `configs/datasets/humman/cam1-view23.yaml`
        - script: `dataset_io/humman.py`


2. run the script with specifying the configuration file:

    ```
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --cfg configs/datasets/zju_mocap/387/cam1-view23.yaml
    ```