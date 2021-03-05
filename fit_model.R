fit_model <- function(x_raw= NULL, 
                      y_raw = NULL) {
  
  
  start_time <- format(Sys.time(), "%Y_%m_%d_%Hh%Mm%Ss")
  # Model training function: given training data (X_raw, y_raw), train this pricing model.
  
  # Parameters
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.
  # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
  # 	A one dimensional array, with values either 0 (most entries) or >0.
  
  # Returns
  # self: (optional), this instance of the fitted model.
  
  tree_method = "gpu_hist" #"hist"
  source("fit_and_evaluate_model.R")
  #source("preprocess_X_data.R")
  library(tidyverse)
  library(tidymodels)
  library(xgboost)
  library(MLmetrics)
  library(mlrMBO)  # for bayesian optimisation  
  require("DiceKriging") # mlrmbo requires this
  require("rgenoud") # mlrmbo requires this
  
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
    folds <- read_csv("output/folds.csv")
  }
  
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
  
  # find the most promising model with  random grid  ----
  
  # I love these parameters and I always want to have them around.  
  simon_params <- data.frame(
    #booster = "gbtree",
    #objective = "reg:tweedie",
    #eval_metric = "rmse",
    #tweedie_variance_power = 1.66,
    gamma = 10,
    max_depth = 3,
    eta = 0.03,
    min_child_weight = 25,
    subsample = 0.6,
    colsample_bytree = 0.3,
    nrounds = 3000
    #tree_method = tree_method
  ) %>% 
    as_tibble()
  
  
  # generate 20 random models parameter sets
  how_many_models <- 4
  set.seed(42)
  gamma <-            data.frame(gamma =c(rep(0,how_many_models/4), runif(3*how_many_models/4)*10)) # 0 à 20
  eta <-              data.frame(eta =c(rep(0.1,how_many_models))) # 0 à 10
  nrounds <-              data.frame(nrounds =c(rep(200,how_many_models))) # 0 à 10
  max_depth <-        data.frame(max_depth = floor(runif(how_many_models)*11 ) + 3)  # 1 à 10
  min_child_weight <- data.frame(min_child_weight = floor(runif(how_many_models) * 100) + 1) # 1 à 100
  subsample <-        data.frame(subsample =runif(how_many_models) * 0.8 + 0.2) # 0.2 à 1
  colsample_bytree <- data.frame(colsample_bytree =runif(how_many_models) * 0.8 + 0.2)  # 0.2 à 1
  random_grid <-gamma %>%
    bind_cols(eta ) %>%
    bind_cols(nrounds ) %>%
    bind_cols(max_depth ) %>%
    bind_cols(min_child_weight) %>%
    bind_cols(subsample) %>%
    bind_cols(colsample_bytree )   %>%
    as_tibble()
  # combine random and hardcoded parameters
  df.params <- simon_params %>%  bind_rows(random_grid) %>%
    mutate(rownum = row_number(),
           rownumber = row_number())
  list_of_param_sets <- df.params %>% 
    group_nest(rownum)  %>% 
    mutate(xgb_params = map(data, 
                            ~  list(
                              booster = "gbtree",
                              eta = .x$eta,
                              max_depth = .x$max_depth,
                              min_child_weight = .x$min_child_weight,
                              gamma = .x$gamma,
                              subsample = .x$subsample,
                              colsample_bytree = .x$colsample_bytree,
                              objective = 'reg:tweedie', 
                              eval_metric = "rmse",
                              tweedie_variance_power = 1.66,
                              tree_method = tree_method
                            )
    )
    )

  
  # the get_random_grid_results tells us which parameter set has the best out-of-fold performance 
  get_random_grid_results <- function(list_of_param_sets){
    
    random_grid_results <- list_of_param_sets %>% 
      mutate(booster = map2(data, xgb_params, function(X, Y){
        message(paste0("model #",       X$rownumber,
                       " eta = ",              X$eta,
                       " max.depth = ",        X$max_depth,
                       " min_child_weigth = ", X$min_child_weight,
                       " subsample = ",        X$subsample,
                       " colsample_bytree = ", X$colsample_bytree,
                       " gamma = ",            X$gamma,
                       " nrounds = ",            X$nrounds
        )
        )

        model_metrics <- fit_and_evaluate_model(
          folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
          model_name = paste0(X$rownumber),
          xgb_params = Y
        )
        

        
        message(paste0("Model_name =", paste0(X$rownumber), " test_gini: ", model_metrics$test_gini, " test_rmse", model_metrics$test_rmse ))
        return(model_metrics)})
        )
      
    
  }
  
  
  tictoc::tic() #34s
  random_grid_results <- get_random_grid_results(list_of_param_sets)
  random_grid_results <- random_grid_results %>% mutate(test_rmse = map_dbl(booster, ~ .x$test_rmse), test_gini = map_dbl(booster, ~ .x$test_gini))
  tictoc::toc()
  write_rds(random_grid_results, paste0("output/", start_time, "_random_grid_results.rds"))
  
  # est-ce que je peux retrouver le gini avec les estimates? yes!
  #test_pour_gini <- bind_rows(folded_data$test)  %>% bind_cols(tibble(estimate = test$booster[[1]]$estimate[[1]])) 
  #MLmetrics::NormalizedGini(test_pour_gini$estimate, test_pour_gini$claim_amount)
  #0.3372843
  

  
  
  # model_metrics <- fit_and_evaluate_model(
  #   folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
  #   model_name = "default",
  #   xgb_params = list(
  #     booster = "gbtree",
  #     objective = "reg:tweedie",
  #     eval_metric = "rmse",
  #     tweedie_variance_power = 1.5,
  #     gamma = 0,
  #     max_depth = 4,
  #     eta = 0.1,
  #     min_child_weight = 5,
  #     subsample = 0.6,
  #     colsample_bytree = 0.6,
  #     tree_method = "hist")
  # )
  
  ## ok we decide that we like these xgb_params best.  Let's fit the most promising model once and for all
  # fit most promising model ----
  
  best_xgb_params <- random_grid_results %>% arrange(-test_gini) %>% head(1) %>% pull(xgb_params) %>% .[[1]]
  # 
  # best_xgb_params <- list(
  #   booster = "gbtree",
  #   objective = "reg:tweedie",
  #   eval_metric = "rmse",
  #   tweedie_variance_power = 1.5,
  #   gamma = 0,
  #   max_depth = 4,
  #   eta = 0.1,
  #   min_child_weight = 5,
  #   subsample = 0.6,
  #   colsample_bytree = 0.6,
  #   tree_method = "hist")
  
  
  trained_recipe <- train_my_recipe(training)
  baked_training <- bake(trained_recipe, new_data = training)
  feature_names =  baked_training %>% select(-claim_amount, -year) %>% colnames()
  xgtrain <-    xgb.DMatrix(
    as.matrix(baked_training %>% select(all_of(feature_names))),
    label = baked_training %>% pull(claim_amount)
  )
  
  xgcv <- xgb.cv(
    params = best_xgb_params,
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
    params = best_xgb_params,
    nrounds = xgcv_best_iteration
  )
  
  my_importance <- xgb.importance(
    feature_names = feature_names,
    model = xgmodel
  ) %>%
    as_tibble()
  
  
  
  
  write_csv(my_importance, paste0("output/",start_time, "_importance.csv"))
  write_csv(as_tibble(best_xgb_params),  paste0("output/", start_time, "_xgb_params.csv"))
  write_rds(trained_recipe, paste0("output/", start_time, "_recipe.rds"))
  xgb.save(xgmodel, fname=paste0("output/", start_time, "_xgmodel.xgb"))
  # calculate adjustment so that we submit an illegal model that loses money on the training set
  preds <- predict(xgmodel, xgtrain)
  claims <- training$claim_amount
  
  ajusts <- tibble(ajusts = sum(claims) / sum(preds))
  write_csv(ajusts,paste0("output/", start_time, "_ajusts.csv"))
  
  return(list(xgmodel = xgmodel, ajusts = ajusts, trained_recipe = trained_recipe)) # return(trained_model)
  
  
}