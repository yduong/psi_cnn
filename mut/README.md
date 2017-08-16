# Structures

## data
contains data file (not commit due to large file size)

## helper
common functions

## ipython
- preprocess: extract X, Y from data files. preprocess data (clean, normalize, encode). Save X,Y into .h5 files to support fit generator
- model: build model to predict data logit value of PSI from synthetic data (A5SS). Use fit_generator API of Keras to support large data size.
The trained model is saved to file for further processing
- prediction: extract submodels from the model built in the previous step. Use the submodel to predict mutation. Plot chart to compare the values predicted using our model
vs HAL model

## results (not commit)
contain intermediate files
