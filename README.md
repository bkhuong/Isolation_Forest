# Isolation_Forest
Implementation of the Isolation Forest Algorithm following the paper by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. See original paper [here](https://github.com/bkhuong/Isolation_Forest/blob/master/IsolationForestPaper.pdf)

## Getting Started 

This repo is structured as follows: 
 - **data**: This is where the training data will be read from. Sample datasets have been provided. 
 - **trained_model**: This is where the trained iForest model and predictions will be saved to. 
 - **iforest.py**: The main iForest script 

### Training 

To train the iForest model as specified in the paper, run:
```
python iforest.py train x.csv -y y.csv
```

Where `x.csv` contains the features to be trained on and `y.csv` is a single column with true labels. See example files in data directory. 

To train an improved version of the iForest model, run: 

```
python iforest.py train x.csv -y y.csv -i 
```

Additionally, the following command-line arguments can be used with the iforest.py script: 
- `t`: to specify the number of trees to build the iForest. The default is 300. 
- `s`: to specify the sample size to build each isolation tree within. This also sets the height limit for each tree as well. The default value is 256 which has been suggested by the isolation forest paper. 
- `i`: to run an improved version of the iforest algorithm. Fit times will increase slightly 
- `n`: filename for saved models, default is `fitted_iForest`

## Prediction 

To used a pre-trained iForest on new data, run: 

```
python iforest.py predict x.csv
```

If multiple pre-trained models exist, specify the filename: 

```
python iforest.py predict x.csv -n model_name.pkl
```

Predictions and scores will be saved in `/trained_model/model_name_predictions.csv`

## Example Run 

**Improved Training Run:**
```
python iforest.py train http_X.csv -Y http_Y.csv -i
```

**Output:**
```
INFO fit time 0.54s
INFO 11940 total nodes in 300 trees
INFO score time 1.08s
300 trees at desired TPR 80.0% getting FPR 0.0072%
Saving trained model to trained_model directory..
```

**Prediction:**
```
python iforest.py predict http_X.csv 
```

**Output:**
```
Saving predictions to trained_model/fitted_iForest_predictions.csv...
```
