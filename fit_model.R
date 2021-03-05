fit_model <- function(x_raw= NULL, 
                      y_raw = NULL, 
                      folds = NULL, 
                      model_name = "default",
                      xgb_params = list(
                        booster = "gbtree",
                        objective = "reg:tweedie",
                        eval_metric = "rmse",
                        tweedie_variance_power = 1.5,
                        gamma = 0,
                        max_depth = 4,
                        eta = 0.1,
                        min_child_weight = 5,
                        subsample = 0.6,
                        colsample_bytree = 0.6,
                        tree_method = "hist"
                      )
) {
  # Model training function: given training data (X_raw, y_raw), train this pricing model.
  
  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.
  # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
  # 	A one dimensional array, with values either 0 (most entries) or >0.
  
  # Returns
  # -------
  # self: (optional), this instance of the fitted model.
  
  
  
  #source("preprocess_X_data.R")
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(MLmetrics)
  
  if(is.null(x_raw)){
    training <- read_csv("training_data.csv")
  } else {
    training <- bind_cols(x_raw, y_raw)
  } 
  
  # We are going to set up a system that allows us to use 100% of the training set for training and testing.  
  # This is important because we don'T have that much data, so keeping only 20% of that is likely to be volatile.  
  
  
  ### create 5 folds  ----
  if(is.null(folds)){
    # set.seed(42)
    # policynumbers <-  training %>% distinct(id_policy) # everyone has 4 records, always put someone's records in the same fold
    # folds <- policynumbers[sample(nrow(policynumbers)),] %>%  # randomize policynumber order, then assign  folds according to row number
    #   mutate(fold = row_number() %% 5 +1) 
    # write_csv(folds, "output/folds.csv")
    read_csv("output/folds.csv")
  }
  set.seed(42)
  # each folds gives me a train and a test  
  folded_data <- tibble(fold = c(1:5)) %>%
    mutate(
      train = map(fold, ~ training %>% inner_join(folds %>% filter(fold != .x))),
      test = map(fold, ~ training %>% inner_join(folds %>% filter(fold == .x))),
    )
  
  ## define recipe, train it for all folds  ----
  
  train_my_recipe <- function(.data){
    my_first_recipe <-
      recipes::recipe(
        claim_amount ~ .,
        .data[0, ]
      ) %>%
      recipes::step_mutate(
        age_when_licensed = drv_age1 - drv_age_lic1,
      ) %>%
      recipes::step_string2factor(recipes::all_nominal()) %>%
      recipes::step_other(recipes::all_nominal(), threshold = 0.05) %>% ##  categories with less than 5% of total are grouped in the "other" group
      recipes::step_dummy(all_nominal(), one_hot = TRUE) %>% # one-hot encode categories
      step_rm(contains("id_policy")) # remove ID
    
    prepped_first_recipe <- recipes::prep(my_first_recipe, .data, retain = FALSE)
    
    return(prepped_first_recipe)
  }
  
  # train recipe for all folds (some may have different dummies for models and cities than the others than the others.)
  folded_data_recipe <-
    folded_data %>%
    mutate(
      trained_recipe =  map(train, ~train_my_recipe( .x)),
      baked_train = map2(train, trained_recipe, ~ recipes::bake(.y, new_data = .x)),
      baked_test = map2(test, trained_recipe, ~ recipes::bake(.y, new_data = .x)),
      feature_names =  map(baked_train, ~ .x %>% select(-claim_amount, -year) %>% colnames())
    )
  
  
  
  folded_data_recipe_xgbmatrix <-
    folded_data_recipe %>% 
    mutate(
      xgtrain = 
        pmap(
          list(baked_train,  feature_names),
          function(.data, .features){
            xgtrain <- xgb.DMatrix(
              as.matrix(.data %>% select(all_of(.features))),
              label = .data %>% pull(claim_amount)
            )
          }
        ),
      xgtest = 
        pmap(
          list(baked_test,  feature_names),
          function(.data, .features){
            xgtrain <- xgb.DMatrix(
              as.matrix(.data %>% select(all_of(.features))),
              label = .data %>% pull(claim_amount)
            )
          }
        )
    )
  
  folded_data_recipe_xgbmatrix_xgbcv <-
    folded_data_recipe_xgbmatrix %>%
    mutate(
      xgcv = map(xgtrain,
                 ~xgb.cv(
                   params = xgb_params,
                   data = .x,
                   nround = 200,
                   nfold = 5,
                   showsd = TRUE,
                   early_stopping_round = 10
                 )
      ),
      xgcv_best_test_rmse_mean = map_dbl(xgcv, ~ .x$evaluation_log$test_rmse_mean[.x$best_iteration]),
      xgcv_best_iteration =  map_dbl(xgcv, ~ .x$best_iteration),
      xgmodel = map2(xgtrain, xgcv_best_iteration,
                     ~ xgboost::xgb.train(
                       data = .x,
                       params = xgb_params,
                       nrounds = .y
                     )
      )
    )
  
  folded_data_recipe_xgbmatrix_xgbcv_metrics <- 
    folded_data_recipe_xgbmatrix_xgbcv %>%
    mutate(
      truth = map(test , ~ .x %>% pull(claim_amount)),
      estimate = map2(xgtest, xgmodel, ~ predict(.y, newdata = .x)),
      test_rmse = map2_dbl(truth, estimate, ~ rmse_vec(.x, .y)),
      test_gini = map2_dbl(truth, estimate, ~ NormalizedGini(.y, .x))
    )
  
  
  
  test_w_preds <- bind_rows(folded_data_recipe_xgbmatrix_xgbcv_metrics$test) %>%
    bind_cols(
      tibble (
        truth = unlist(folded_data_recipe_xgbmatrix_xgbcv_metrics$truth),
        estimate = unlist(folded_data_recipe_xgbmatrix_xgbcv_metrics$estimate)
      )
    )
  
  
  model_metrics <-
    tibble(
      model_name = model_name,
      test_rmse = rmse_vec(test_w_preds$truth, test_w_preds$estimate),
      test_gini = NormalizedGini(test_w_preds$estimate, test_w_preds$truth),
      mean_xgcv_best_iteration = ceiling(mean(folded_data_recipe_xgbmatrix_xgbcv_metrics$xgcv_best_iteration)),
      xgcv_best_test_rmse_means = list(folded_data_recipe_xgbmatrix_xgbcv_metrics$xgcv_best_test_rmse_mean),
      xgcv_best_iterations = list(folded_data_recipe_xgbmatrix_xgbcv_metrics$xgcv_best_iteration),  
      test_rmses = list(folded_data_recipe_xgbmatrix_xgbcv_metrics$test_rmse),
      test_ginis = list(folded_data_recipe_xgbmatrix_xgbcv_metrics$test_gini),
      params = list(xgb_params)
      
    )
  write_rds(model_metrics, paste0("output/",model_name, "_model_metrics.rds"))
  write_csv(model_metrics %>% select(model_name, test_rmse, test_gini, mean_xgcv_best_iteration), paste0("output/",model_name, "_model_metrics.csv"))
  
  
  # WE have gone through all this to estimate out of fold performance using our whole data set as a "test" dataset.  Now  let's fit on full training  data
  
  
  trained_recipe <- train_my_recipe(training)
  baked_training <- bake(trained_recipe, new_data = training)
  feature_names =  baked_training %>% select(-claim_amount, -year) %>% colnames()
  xgtrain <-    xgb.DMatrix(
          as.matrix(baked_training %>% select(all_of(feature_names))),
          label = baked_training %>% pull(claim_amount)
        )
  xgcv <- xgb.cv(
    params = xgb_params,
    data = xgtrain,
    nround = 200,
    nfold = 5,
    showsd = TRUE,
    early_stopping_round = 10
  )
  xgcv_best_test_rmse_mean = xgcv$evaluation_log$test_rmse_mean[xgcv$best_iteration]
  xgcv_best_iteration =  xgcv$best_iteration
  xgmodel = xgboost::xgb.train(
                   data = xgtrain,
                   params = xgb_params,
                   nrounds = xgcv_best_iteration
                 )
  
my_importance <- xgb.importance(
    feature_names = feature_names,
    model = xgmodel
  ) %>%
    as_tibble()
  
  write_csv(my_importance, paste0("output/", model_name, "_importance.csv"))
  write_csv(as_tibble(xgb_params),  paste0("output/", model_name, "_xgb_params.csv"))
  
  # calculate adjustment so that we submit an illegal model that loses money on the training set
  preds <- predict(xgmodel, xgtrain)
  claims <- training$claim_amount
  
  ajusts <- sum(claims) / sum(preds)
  write_rds(ajusts,paste0("output/", model_name, "_ajusts.csv"))
  xgb.save(xgmodel, fname=paste0("output/", model_name, "_xgmodel.xgb"))
  return(xgmodel) # return(trained_model)
}
