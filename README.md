# Indoors-Traversability-Estimation-with-RGB-Laser-Fusion

# Indoors-Traversability-Estimation-with-Less-Labels for Mobile Robots


1. Open a terminal window
2. Clone the repo

## Install the requirements on your machine 

`pip install -r requirements.txt`

## Fine-tuning on your dataset 


For fine-tuning on your dataset:
1. Open rgb_fusion.py
2. Specify the train and test dataset image paths as '/home/../../set'
3. Specify the laser path as '/home/../../set'
3. Run `python3 rgb_fusion.py`

You can follow the exact same process for fine-tuning with any other RGB combination you wish



### Dataset

Link to the dataset [here](https://drive.google.com/file/d/1W2kK7GgNg8mCvbms-SRUnWsQ3FSVoDbu/view?usp=sharing)
To annotate the dataset according to the PU method, simply run annotate.py on your image/laser dataset.
