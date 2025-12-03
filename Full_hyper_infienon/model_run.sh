#!/bin/bash

time python run.py \
  --test_size 0.2 \
  --select_model Decision_Tree \
  --table_name infineon_gesture \
  --limit_num 2500 \
  --max_depth_dtc None \
  --min_samples_split 2 \
  --min_samples_leaf 1 \
  --max_features_dtc None \
  --criterion gini \
  --max_depth_rf None \
  --n_estimators_rf 100 \
  --max_features_rf sqrt \
  --n_estimators_gb 100 \
  --max_depth_gb 3 \
  --learning_rate 0.1 \
  --subsample 1.0


# python run.py --test_size 0.2 --select_model Decision_Tree --table_name infineon_gesture --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion gini --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0
