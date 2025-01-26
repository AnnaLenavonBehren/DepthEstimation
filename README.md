# Depth Estimation

This project includes the implementation of different approaches to 3D gaze distance estimation:  

### Center and Vergence

Serving as an experimental baseline, this notebook includes the vergence-based methods by Wang et al. [1] and Weier et al. [2] alongside the depth at the current gaze sample as a trivial estimate. 

### SVR

This notebook implements a Support Vector Regression with radial basis functions closely following the implementation of Weier et al. [2]. It evaluates different loss functions and the effects of training the model on unseen participants versus all training data combined.

----

### MLP and CNNs

The following models are part of a larger Python project ([Models/](https://github.com/AnnaLenavonBehren/DepthEstimation/tree/976d27277c6ec829c67aed71650ee333602d63db/Models)), that tests an MLP and various different CNN architectures with different training data and parameters. It also has the option of testing the performance when the features are permuted.  
The options for training can be set in the main file; feature permutation requires changing the eye-tracking file in the corresponding data file ([Models/data/](https://github.com/AnnaLenavonBehren/DepthEstimation/tree/976d27277c6ec829c67aed71650ee333602d63db/Models/data)).  

For already trained models, the [weights](https://github.com/AnnaLenavonBehren/DepthEstimation/tree/976d27277c6ec829c67aed71650ee333602d63db/Models/saved_models) and the [resulting predictions](https://github.com/AnnaLenavonBehren/DepthEstimation/tree/976d27277c6ec829c67aed71650ee333602d63db/Models/results) are available.


Running the project requires the raw depth and eye-tracking data collected within the Eye-tracking Study. The data is not uploaded but can be made available upon request.


----


**References:**


[1] R. I. Wang, B. Pelfrey, A. T. Duchowski and D. H. House, _Online Gaze Disparity via Bioncular Eye Tracking on Stereoscopic Displays_, 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, Zurich, Switzerland, 2012, pp. 184-191, doi: 10.1109/3DIMPVT.2012.37.  
[2] Martin Weier, Thorsten Roth, André Hinkenjann, and Philipp Slusallek. 2018. _Predicting the gaze depth in head-mounted displays using multiple feature regression_. In Proceedings of the 2018 ACM Symposium on Eye Tracking Research \&; Applications (ETRA '18). Association for Computing Machinery, New York, NY, USA, Article 19, 1–9. https://doi.org/10.1145/3204493.3204547