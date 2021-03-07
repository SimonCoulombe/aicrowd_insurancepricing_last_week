
#!/bin/bash

export DATASET_PATH=training_data.csv

Rscript predict.R

WEEKLY_EVALUATION=true Rscript predict.R
