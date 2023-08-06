# LFASR-ELFR
This project provides the code for 'Enhanced Light Field Reconstruction by Combining Disparity and Texture Information in PSVs via Disparity-Guided Fusion', IEEE TCI, 2023. [paper link](https://ieeexplore.ieee.org/document/10158790)

## Framework Overview
<div align=center>
<img src="https://github.com/GilbertRC/LFASR-ELFR/blob/main/Figs/Framework.png">
</div>

Note: The explicit-depth-based and implicit-depth-based pipelines adopt the basic structure of [GA-Net](https://github.com/jingjin25/LFASR-geometry) and [MALFRNet](https://ieeexplore.ieee.org/document/9258385) (*w/o* their refinement), respectively.

## Requirements
- Python=3.7  
- PyTorch=1.8.0  
- scikit-image=0.14.2
- Matlab (for .h5 file generation)

## Dataset
1. Download the [TrainingSet](https://pan.baidu.com/s/1COZrlPgPcbyyp3737k2OCA) (code: 3f2x) and [TestSet](https://pan.baidu.com/s/1mvp954aeONOSmmKeOzq8og) (code: 6c31) and put them under './LFData/' folder.
2. Run `PrepareData_xxx.m` to generate .h5 file for training and test.
3. Or directly download our generated [.h5 file](https://pan.baidu.com/s/1JSAdFA2FPirndJ6HOOOGmQ) (code: sgca).

## Training
**model_HCI for synthetic datasets, 2x2&rarr;7x7 interpolation**
```
python train_HCI.py --train_dataset HCI --disp_range 4 --num_planes 50 --angular_in 2 --angular_out 7 --epoch 50000 --learning_rate 1e-4 --decay_rate 0.5 --decay_epoch 5000 --batch_size 1 --patch_size 64
```
**model_SIG for real-world datasets, 2x2&rarr;7x7 interpolation**
```
python train.py --train_dataset SIG --disp_range 1.5 --num_planes 32 --angular_in 2 --angular_out 7 --epoch 10000 --learning_rate 1e-4 --decay_rate 0.5 --decay_epoch 1000 --batch_size 1 --patch_size 64
```

The training curve for *disp_thres* appears like:
<div align=left>
<img src="https://github.com/GilbertRC/LFASR-ELFR/blob/main/Figs/curve.png">
</div>

## Test using pre-trained model
**Synthetic datasets (*HCI, HCI old and Inria DLFD*), 2x2&rarr;7x7 interpolation**
```
python test_HCI.py --model_dir pretrained_model --train_dataset HCI --disp_range 4 --num_planes 50 --angular_in 2 --angular_out 7 --input_ind 0 6 42 48 --crop 1
```
**Real-world datasets (*30scenes, occlusions and reflective*), 2x2&rarr;7x7 interpolation**
```
python test.py --model_dir pretrained_model --train_dataset SIG --disp_range 1.5 --num_planes 32 --angular_in 2 --angular_out 7 --input_ind 0 6 42 48 --crop 0
```

## Performance
Our performance under the 2x2&rarr;7x7 interpolation task:
<div align=center>
  <img src="https://github.com/GilbertRC/LFASR-ELFR/blob/main/Figs/Table1.png">
</div>

## Citation
```
@ARTICLE{LFASR-ELFR,  
  title={Enhanced Light Field Reconstruction by Combining Disparity and Texture Information in PSVs via Disparity-Guided Fusion},
  author={Yilei Chen and Xinpeng Huang and Ping An and Qiang Wu},
  journal={IEEE Transactions on Computational Imaging},
  year={2023},
  volume={9},
  pages={665-677}
  month={Jul.}}            
```

Any questions regarding this work can contact yileichen@shu.edu.cn.
