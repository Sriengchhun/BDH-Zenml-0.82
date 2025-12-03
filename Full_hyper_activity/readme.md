# Run the model with argument


```
python run.py --test_size 0.2 --select_model Decision_Tree --table_name infineon_gesture --limit_num 25000  --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 0.9 --max_features_dtc None --max_depth_rf None --n_estimators_rf 100 --max_depth_gb 3 --learning_rate 0.1  --criterion gini

```


```
python run.py --test_size 0.2 --select_model Decision_Tree --table_name infineon_gesture --limit_num 25000  --max_depth_dtc None --min_samples_split 0.2  --max_depth_rf None --n_estimators_rf 100 --max_depth_gb 3 --learning_rate 0.1  --criterion gini

```


python run.py --test_size 0.2 --select_model Decision_Tree --table_name infineon_gesture --limit_num 25000  --max_depth_dtc None --max_depth_rf None --n_estimators_rf 100 --max_depth_gb 3 --learning_rate 0.1  --criterion gini




## Hyper parameter:

1) test_size=test_size, 
2) selected_model=select_model, 
3) table_name=table_name, 
4) limit_num=limit_num, 
5) max_depth_dtc=max_depth_dtc,
6) # min_samples_split=min_samples_split,
7) # min_samples_leaf=min_samples_leaf,
8) max_depth_rf=max_depth_rf,
9) n_estimators_rf=n_estimators_rf,
10) max_depth_gb=max_depth_gb,
11) learning_rate=learning_rate,
12) criterion=criterion

min_samples_split and min_samples_split (are use the same default of this 3 models) 



### Run model for Decision Tree
```
python run.py --test_size 0.2 --select_model Decision_Tree --table_name applewatch --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion gini --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0 

```

### Run model for Randomforest

```
python run.py --test_size 0.2 --select_model Random_Forest --table_name applewatch --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion gini --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0

```



### Run model for Gradient Boosting
- criterion = squared_error
```
python run.py --test_size 0.2 --select_model Gradient_Boosting --table_name applewatch --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion squared_error --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0 

```

- criterion = friedman_mse
```
python run.py --test_size 0.2 --select_model Gradient_Boosting --table_name applewatch --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion friedman_mse --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0 --criterion_gb squared_error

```

criterion {'squared_error', 'friedman_mse'}



time python run.py --test_size 0.2 --select_model Gradient_Boosting --table_name wesafe_ais --limit_num 2500 --max_depth_dtc None --min_samples_split 2 --min_samples_leaf 1 --max_features_dtc None --criterion friedman_mse --max_depth_rf None --n_estimators_rf 100 --max_features_rf sqrt --n_estimators_gb 100 --max_depth_gb 3 --learning_rate 0.1 --subsample 1.0 --criterion_gb squared_error