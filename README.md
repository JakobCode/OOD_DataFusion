## Handling Unexpected Inputs - <br>Incorporating Source-Wise Out-of-Distribution Detection into SAR-Optical Data Fusion for Scene Classification


This repository comes a along with the paper <a href="TBÀ" target="_blank" rel="noreferrer noopener">"Handling Unexpected Inputs - Incorporating Source-Wise Out-of-Distribution Detection into SAR-Optical Data Fusion for Scene Classification"</a> published in <i>TBA</i>. 

### Overview
This work proposses the incorporation of out-of-distribution detectors into a Sentinel-1 and Sentinel-2 data fusion neural network for multilabel scene classification. The resulting approach is less sensitive to changes in data distributions of individual data sources, as for example occurs when (unknown) clouds cover the optical sample. The experiments in this work are based on the multimodal version of the <a href="https://bigearth.net/" target="_blank" rel="noreferrer noopener">BigEarthNet Data Set</a> and all visualizations of samples are taken from this data set.
<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/216149189-a16bec91-26f0-468f-acdd-a12e01f8855d.png">
</p>

### Prerequisites
This repository has been tested under ```Python 3.9.12``` in a *unix* development environment. <br> 
For a setup, clone the repository and ``cd``to the root of it. <br>
Create a new environment and activate it, for instance, on unix via
```
python -m venv venv && source venv/bin/activate
```
Then install the needed packages via:
```
pip install --upgrade pip
pip install -r requirements.txt
```

### Experiment Setup
In order to train and test the proposed fusion network and the addtional out-of-distribution detectors a new Experiment containing a config file should be setup. An example and template config file can be found in `./Experiments/Experiment1`. Local paths in this examples need to be the same as the example config or adjusted in the config file before running experiments. 

Besides this repository the <a href="https://bigearth.net/" target="_blank" rel="noreferrer noopener">BigEarthNet-MM Data Set</a> needs to be downloaded and the two folders `BigEarthNet-S1-v1.0`(Sentinel-1) and `BigEarthNet-v1.0` (Sentinel-2) need to be placed in one common folder, the `data_root_path`. Adjust the `data_root_path` in the configuration file. 

### Data Preprocessing
The BigEarthNet data set contains separate '.tif'-files for each Sentinel-2 band and Sentinel-1 polarization. As a first step, we merge these bands into one single file. This takes a while and can be either done automatically when using the data loader for the first time or run separately by calling the `
main` method in `BigEarthNet_DataLoader.py` via
```
python ./code/data/bigearthnet_dataloader.py -cfg path_to_cfg
```
Important: The data preparation can take up to multiple hours for the full data set but only has to be run ones. 
### Training Data Fusion Network 
The data fusion network contains two input branches extracting modality specific features from the SAR and the optical modality. The combined part is trained with multiple forward passes of different combinations of SAR and optical features and zeros. 

In order to train the network call
```
python ./code/train_network.py -cfg path_to_cfg
```

<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/216146250-1cc6160e-e3cc-40e2-984f-7a816486c789.png" width="48%">
</p>

### Training Out-of-Distribution Detectors
The out-of-distribution detectors are trained on the pre-trained feature extractors given by the optical and the SAR branch of the fusion network. To run the training of the ood-detectors run
```
python ./code/train_ood.py -cfg path_to_cfg
```
<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/216146247-2e54932e-3fcb-42b4-b355-036117b59b69.png" width="48%">
</p>

### Experimental Setup
For the testing we considered the following scenarios: 
<ul>
  <li>Official BigEarthNet test split.</li>
  <li>Cloudy samples which are supplement to the BigEarthNet data set.</li>
  <li>A handpicked subset of <it>very cloudy</it> samples of the cloudy set.</li>
  <li>Ice and Snow samples which are supplement to the BigEarthNet data set.</li>
  <li>Corrupted optical samples (random gausian pixels from training set statistics).</li>
  <li>Corrupted SAR samples (random gausian pixels from training set statistics).</li>
  <li>Missing optical samples.</li>
  <li>Missing SAR samples.</li>
</ul> 
<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/219948968-0361ee24-3956-4b14-96ee-3d76f0689f16.png">
</p>

### Performance Testing
For testing run the test script with one of the eight scenarios listed above: `train`, `val`, `test`,  `ood_cloudy`,  `ood_cloudy_handpicked`,  `ood_ice_and_snow`,  `left_out_classes`,  `corrupted_sar`, `corrupted_opt`, `missing_sar`,  `missing_opt`:
```
python ./code/test_network.py -cfg path_to_config -s seed_value -ts test_scenario
```
<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/216149187-4c96da94-e478-4a64-b70b-f5a4b9bf9894.png">
</p>

### Citation
If you find our code or results useful for your research, please consider citing: 
```
@article{gawlikowski2023handling,
  title={Handling Unexpected Inputs - <br>Incorporating Source-Wise Out-of-Distribution Detection into SAR-Optical Data Fusion for Scene Classification},
  author={Gawlikowski, Jakob and Saha, Sudipan and Niebling, Julia and Zhu, Xiao Xiang},
  journal={TBA},
  year={TBA}
}
```
