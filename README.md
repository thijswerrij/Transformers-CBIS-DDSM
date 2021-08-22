# Classifying mammograms using visual transformers

## Important files
* [pretrained_ResViT.py](pretrained_ResViT.py "pretrained_ResViT.py") - Run a combination of a pretrained ResNet-18 model and visual transformer modules, based on Wu et al. [[1]](#1), on CBIS-DDSM data [[2]](#2).
* [resnet.py](resnet.py "resnet.py") - Run a pretrained ResNet-18 model on CBIS-DDSM data [[2]](#2).
* [cbis_ddsm_train.py](cbis_ddsm_train.py "cbis_ddsm_train.py") - CBIS-DDSM data loading and helper functions (used by other files).
* [load_data.py](load_data.py "load_data.py") - Cropping, downscaling and saving CBIS-DDSM data to hdf5 format. Necessary before running any experiments.

## Acknowledgements
This repository uses code from two other repositories:
* Large part of this project was based on code by Md Tahmid Hossain (tahmid0007), see [VisualTransformers](https://github.com/tahmid0007/VisualTransformers).
* During cropping we use [crop_mammogram.py](https://github.com/nyukat/breast_cancer_classifier/blob/master/src/cropping/crop_mammogram.py) from the [breast_cancer_classifier](https://github.com/nyukat/breast_cancer_classifier) repository.
## References

<a id="1">[1]</a> Wu et al. (2020) Visual Transformers: Token-based Image Representation and Processing for Computer Vision https://arxiv.org/abs/2006.03677

<a id="2">[2]</a> Rebecca Sawyer Lee et al. (2016). Curated Breast Imaging Subset of DDSM [Dataset]. The Cancer Imaging Archive. DOI: [https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY](http://dx.doi.org/10.7937/K9/TCIA.2016.7O02S9CY)