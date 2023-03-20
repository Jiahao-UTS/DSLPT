#Dynamic Sparse Local Patch Transformer

PyTorch training code and pretrained models for **DSLPT** (Dynamic Sparse Local Patch Transformer).

<img src='Figures/Fig1.jpg' width="450px">

## Installation
#### Note: this released version was tested on Python3.8 and Pytorch 1.10.2.

Install system requirements:
```
sudo apt-get install python3-dev python3-pip python3-tk libglib2.0-0
```

Install python dependencies:
```
pip3 install -r requirements.txt
```

## Run training code on WFLW dataset
1. Download and process WFLW dataset
    * Download WFLW dataset and annotation from [Here](https://wywu.github.io/projects/LAB/WFLW.html).
    * Unzip WFLW dataset and annotations and move files into ```./Data``` directory. Your directory should look like this:
        ```
        DSLPT
        └───Data
           │
           └───WFLW
              │
              └───WFLW_annotations
              │   └───list_98pt_rect_attr_train_test
              │   │
              │   └───list_98pt_test
              └───WFLW_images
                  └───0--Parade
                  │
                  └───...
        ```

2. Download pretrained weight of HRNetW18C 
   * Download pretrained weight of HRNetW18C from [Here](https://github.com/HRNet/HRNet-Image-Classification).
   * Move files into ```./Config``` directory. Your directory should look like this:
        ```
        DSLPT
        └───Config
           │
           └───hrnetv2_w18_imagenet_pretrained.pth
        ```       


3. ```python ./train.py```.

## Run Evaluation on WFLW dataset
1. Download and process WFLW dataset
    * Download WFLW dataset and annotation from [Here](https://wywu.github.io/projects/LAB/WFLW.html).
    * Unzip WFLW dataset and annotations and move files into ```./dataset``` directory. Your directory should look like this:
        ```
        DSLPT
        └───Dataset
           │
           └───WFLW
              │
              └───WFLW_annotations
              │   └───list_98pt_rect_attr_train_test
              │   │
              │   └───list_98pt_test
              └───WFLW_images
                  └───0--Parade
                  │
                  └───...
        ```
2. Download pretrained model from Google Drive.
    * WFLW
   
    <table>
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <td>Model Name</td>
          <td>NME</td>
          <td>FR<sub>0.1</sub></td>
          <td>AUC<sub>0.1</sub></td>
          <td>download link</td>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>DSLPT-6-layers</td>
          <td>4.01</td>
          <td>2.52</td>
          <td>0.607</td>
          <td><a href="https://drive.google.com/file/d/1-1ZMLTBk6D3FxdlMC0sEJ9Fsx301ZUcC/view?usp=sharing">download</a></td>
        </tr>
        <tr>
          <th>2</th>
          <td>DSLPT-12-layers</td>
          <td>3.98</td>
          <td>2.44</td>
          <td>0.609</td>
          <td><a href="https://drive.google.com/file/d/1vuXDOppWhSRLu9Oxrge9RvAeAQr_-ez_/view?usp=sharing">download</a></td>
        </tr>
      </tbody>
    </table>
   
    Put the model in ```./weights``` directory.

3. Test

    ```
    python validate.py --checkpoint=<model_name>
    For example: python validate.py --checkpoint=DSLPT_WFLW_6_layers.pth
    ```
    
    Note: if you want to use the model with 12 layers, you need to change ```_C.TRANSFORMER.NUM_DECODER``` for
    6 to 12 in ```./Config/default.py```.


##Citation
If you find this work or code is helpful in your research, please cite:
```
@ARTICLE{DSLPT,
  title={Robust Face Alignment via Inherent Relation Learning and Uncertainty Estimation},
  author={Jiahao Xia and Min Xu and Haimin Zhang and Jianguo Zhang and Wenjian Huang and Hu Cao and Shiping Wen},
  booktitle={TPAMI},
  year={2023}
}
```

```
@inproceedings{SLPT,
  title={Sparse Local Patch Transformer for Robust Face Alignment and Landmarks},
  author={Jiahao Xia and Weiwei Qu and Wenjian Huang and Jianguo Zhang and Xi Wang and Min Xu},
  booktitle={CVPR},
  year={2022}
}
```

# License
DSLPT is released under the GPL-2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Acknowledgments
   * This repository borrows or partially modifies the models from [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
and [DETR](https://github.com/facebookresearch/detr)
