# Hierarchical-Encoding-and-Text-Guided-3D-Modeling-from-Single-Free-Hand-Sketches
Hierarchical Encoding and Text-Guided 3D Modeling from Single Free-Hand Sketches
<img width="2572" height="1216" alt="image" src="https://github.com/user-attachments/assets/62158b5c-18b0-4468-b356-cfe276c998ad" />
# Hierarchical Sketch Encoding and Text-Guided 3D Modeling: A Novel Approach for Single Free-Hand Sketch Reconstruction

Official PyTorch implementation of paper `Hierarchical Sketch Encoding and Text-Guided 3D Modeling: A Novel Approach for Single Free-Hand Sketch Reconstruction`, submitted to The Visual Computer.

## Environments
- `git clone --recursive [https://github.com/bennyguo/sketch2model](https://github.com/QinchuanLei/HETG-3D-Modeling).git`
- Python>=3.10.15
- PyTorch>=2.5
- install dependencies: `pip install -r requirements.txt`
- build and install Soft Rasterizer: `cd SoftRas; python setup.py install`

## Training
- Download `shapenet-synthetic.zip` [here](https://drive.google.com/drive/folders/1_DKZV6KtqpLKRoBd0JgOgf60wi1LYm6s?usp=sharing), and extract to `load/`:
```
load/
└───shapenet-synthetic /
    │   02691156/
    │   ... ...
    ...
```
- Train on airplane:
```
python train.py --name exp-airplane --class_id 02691156
```

You may specify arguments listed in `options/base_options.py` and `options/train_options.py`. Saved meshes are named with the corresponding (ground truth or predicted) viewpoints in format `e[elevation]a[azimuth]`. `pred` in the filename indicates predicted viewpoint, otherwise the viewpoint is ground truth value (or user-specified when inference, see the Inference section below).

Supported classes:
```
02691156 Airplane
02828884 Bench
02933112 Cabinet
02958343 Car
03001627 Chair
03211117 Display
03636649 Lamp
03691459 Loudspeaker
04090263 Rifle
04256520 Sofa
04379243 Table
04401088 Telephone
04530566 Watercraft
```

## Evaluation
- Test on ShapeNet-Synthetic testset:
```
python test.py --name [experiment name] --class_id [class id] --test_split test
```
- To test on our ShapeNet-Sketch dataset, you need to first download `shapenet-sketch.zip` [here](https://drive.google.com/drive/folders/1_DKZV6KtqpLKRoBd0JgOgf60wi1LYm6s?usp=sharing) and extract to `load/`, then
```
python test.py --name [experiment name] --class_id [class id] --dataset_mode shapenet_sketch --dataset_root load/shapenet-sketch
```
About file structures of our ShapeNet-Sketch dataset, please see the dataset definition in `data/shapenet_sketch_dataset.py`.
## Acknowledgments
Our code is inspired by https://github.com/QinchuanLei/HETG-3D-Modeling
