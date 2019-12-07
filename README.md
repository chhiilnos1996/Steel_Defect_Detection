# ECE 285 Final Project (D): Severstal Steel Defect Detection
## Authors: Chi-Hsin Lo, Subrato Chakravorty, Utkrisht Rajkumar

### Description
This is **project D: Severstal Steel Defect Detection** using semantic segmentation by team **3Nature_2Journal_8GPU**. 

The dataset for this project can be found [here](https://www.kaggle.com/c/severstal-steel-defect-detection/data). To run with our default path setup, create a new folder called "inputs" in the same directory as the one containing this cloned respository. Download and extract the dataset inside "inputs." The kaggle submission kernel can be found [here](https://www.kaggle.com/urajkumar/ece285-project-d). It contains the code to upload submission.csv files and submit to Kaggle. Inside the kernel, load the dataset called `ece285_sub` to access all generated submission files. The submissions can be directly downloaded from kaggle [here](www.kaggle.com/dataset/3c0f2f90f63341b33e3ec48908abd621c72ed0487aaa3a7341b1df17c1930c89).

### Requirements
All the package requirements can be found in requirements.txt. To install the requirements:

```
git clone https://github.com/ucrajkumar/ece285
cd ece285
pip install -r requirements.txt
```

If working in a conda environment, use: 

`conda install --file requirements.txt`

The required file structure to run with our default paths are as follows:
**File Structure**

* Current Directory
  * Inputs
    * Downloaded data from Kaggle
  * ece285 (this repo)
    
### Code organization


file name | Description of file 
--- | ---
Demo.ipynb | Qualitatively evaluate all models on validation and test set images
Figures.ipynb | Recreate figures from report
Train.ipynb | Train u-net, u-net with residual encoder, and u-net with inverted residual encoder
deeplabv3.ipynb | Train deeplabv3+
Inference.ipynb | Predict on test set and export results
data_gen.py | Data generator for loading train, validation, and test images
model.py | Code for building all 4 models
utils.py | Accuracy and loss metrics, conversion from mask to RLE and vice-versa, and post-processing
train_idx.npy | indices for training set
val_idx.npy | indices for validation set
-----------------------------------
folder name | Description of file 
--- | ---
ex_images | Example validation and test images for Demo.ipynb
history | Training history for each model
models | Trained model files
submissions | Exported results on test set
