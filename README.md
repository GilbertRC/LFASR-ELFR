# LFASR-ELFR
This project provides the code for 'Enhanced Light Field Reconstruction by Combining Disparity and Texture Information in PSVs via Disparity-Guided Fusion', IEEE TCI, 2023. [paper link](https://ieeexplore.ieee.org/document/10158790)

## Framework Overview
<div align=center>
<img src="https://github.com/GilbertRC/LFASR-ELFR/blob/main/Figs/Framework.png">
</div>

Note: The explicit-depth-based and implicit-depth-based pipelines adopt the basic structure of [GA-Net](https://github.com/jingjin25/LFASR-geometry) and [MALFRNet](https://ieeexplore.ieee.org/document/9258385) (w/o their refinement), respectively.

## Requirements
- Python=3.7  
- PyTorch=1.8.0  
- scikit-image=0.14.2
- Matlab (for data generation)

## Dataset


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
