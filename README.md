# Autobid LTV estimation

### Data 
We assume that there is a data folder and that the csv files 'ds_test_eval_100k.csv' and 'ds_test_train_1M.csv' 
have been downloaded in the folder data.

### Requirements
- python~=3.9.7
- pandas~=1.2.3
- scikit-learn~=1.0.2

### Code Organisation
The file data_manage.py contains some tools used for manipulating and preprocessing
the data.

The jupyter notebook autobid contains the different steps of the EDA, pre-processing,
model selection, model calibration and testing. All the steps are detailed and
commented in the notebook.

### Analysis overview
We initially perform an EDA to get to know the data better and see what pre-processing steps needs to be done.
We analyse the average results accross the different granularities and the corresponding
features. That analysis suggest that the null values should be filled using the average of the corresponding 
column for revenue features and the average of the corresponding raw for installs features.

To choose the model we perform a study on Ridge regressor, Random Forest and LightGBM
on a sample of the  training data. The model selected is Light GBM.

We perform then an hyperparameter optimization for the selected model and select n_estimators=200 and max_depth=6.

We calibrate the model on all the training data using the selected hyperparameters.

We test the model on the eval data and find a r2 of 0.35

### Further experiments:
- Given the size of the dataset, it would have been interesting to explore Neural Networks. 
- It would have been interested to fine-tune other parameters for the LightGBM
- It could be interesting to see if by reducing the number of features, we could get some improvements:
    - either by grouping some columns and average them
    - or by performing some PCA before training the model

