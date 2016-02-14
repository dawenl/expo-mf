# ExpoMF
This repository contains the source code to reproduce all the experimental results as described in the paper ["Modeling User Exposure in Recommendation"](http://arxiv.org/abs/1510.07025) (WWW'16).

## Dependencies
The python module dependencies are:
- numpy/scipy
- scikit.learn
- joblib
- bottleneck
- pandas (needed to run the example for data preprocessing)

**Note**: The code is mostly written for Python 2.7. For Python 3.x, it is still usable with minor modification. If you run into any problem with Python 3.x, feel free to contact me and I will try to get back to you with a helpful solution.  

## Datasets
- [Taste Profile Subset](http://labrosa.ee.columbia.edu/millionsong/tasteprofile)
- [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html): the pre-processed data that we used in the paper can be downloaded [here](http://dawenl.github.io/data/gowalla_pro.zip).

We also used the arXiv and Mendeley dataset in the paper. However, these datasets are not publicly available. With Taste Profile Subset and Gowalla, we can still cover all the different variations of the model presented in the paper. 

We used the weighted matrix factorization (WMF) implementation in [content_wmf](https://github.com/dawenl/content_wmf) repository. 

## Examples
See example notebooks in `src/`. 
