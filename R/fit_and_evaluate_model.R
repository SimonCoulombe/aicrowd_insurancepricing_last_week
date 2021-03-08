


## OK I did something really overkill with B- randomgrid and C- bayesian.
# bercause I felt I didnt have much data, I turnede my whole dataset into a test set.
# steps:
# create 5 20% folds.
# do xgb.cv on fold A B C  D to find optimal number of iterations
# train model on A B C D with optimal number of iterations
# predict on fold E.  (the holdout folds)

# repeat  4 more times, using the different folds as holdout.
# don't think it was worth it.


# the evaluate_multiple_parameters tells us which parameter set has the best out-of-fold performance
evaluate_multiple_parameters <- function(list_of_param_sets, ... ) {
  random_grid_results <- list_of_param_sets %>%
    mutate(model_metrics = map2(data, xgb_params, function(X, Y) {
      message(paste0(
        "model #", X$rownumber,
        " eta = ", X$eta,
        " max.depth = ", X$max_depth,
        " min_child_weigth = ", X$min_child_weight,
        " subsample = ", X$subsample,
        " colsample_bytree = ", X$colsample_bytree,
        " gamma = ", X$gamma,
        " nrounds = ", X$nrounds
      ))
      
      model_metrics <- fit_and_evaluate_model(
        training = training,
        model_name = paste0(X$rownumber),
        xgb_params = Y, 
        ...
        )
      
      message(paste0("Model_name =", paste0(X$rownumber), " test_gini: ", model_metrics$test_gini, " test_rmse", model_metrics$test_rmse))
      return(model_metrics)
    }))
  
  return(random_grid_results %>%  unnest(cols = model_metrics))
}

fit_and_evaluate_model <- function(training,
                                   model_name,
                                   xgb_params,
                                   recipe_function = train_my_recipe_xgb_kitchensink,
                                   max_rounds = 50,
                                   n_folds_step1 = 5, # create how many folds  at step 1 (train vs test?  
                                   do_folds_step2 = c(1,2),  # c(!,2,3,4,5) how many of the n_folds_step1 do we do xgb.cv on? c(!,2,3,4,5) is better, but takes linger too
                                   n_folds_step2 = 3) { #  when we do xgb.cv how many folds do we break the data into?
# fit_and_evaluate_model(training = training, model_name = "prout", xgb_params = best_xgb_params)
                                       
  submodel_start_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
  # assign all policies to a fold
  folds <- create__group_folds(training, id_policy, n_folds_step1)
  
  # create train and test set for all folds
  folded_data <- tibble(folds %>% distinct(fold) %>% arrange(fold)) %>%
    mutate(
      train = map(fold, ~ training %>% inner_join(folds %>% filter(fold != .x))),
      test = map(fold, ~ training %>% inner_join(folds %>% filter(fold == .x))),
    )
  
  # train recipe for all folds (some may have different dummies for models and cities than the others than the others.)
  folded_data_recipe <-
    folded_data %>%
    mutate(
      trained_recipe = map(train, ~ recipe_function(.x)),
      baked_train = map2(train, trained_recipe, ~ recipes::bake(.y, new_data = .x)),
      baked_test = map2(test, trained_recipe, ~ recipes::bake(.y, new_data = .x)),
      feature_names = map(baked_train, ~ .x %>%
                            select(-claim_amount, -year, -claim_amount_backup) %>%
                            colnames()) # drop year
    )
  
  # create xgb matrix from baked data
  folded_data_recipe_xgbmatrix <-
    folded_data_recipe %>%
    mutate(
      xgtrain =
        pmap(
          list(baked_train, feature_names),
          function(.data, .features) {
            xgtrain <- xgb.DMatrix(
              as.matrix(.data %>% select(all_of(.features))),
              label = .data %>% pull(claim_amount)
            )
          }
        ),
      xgtest =
        pmap(
          list(baked_test, feature_names),
          function(.data, .features) {
            xgtrain <- xgb.DMatrix(
              as.matrix(.data %>% select(all_of(.features))),
              label = .data %>% pull(claim_amount)
            )
          }
        )
    )
  
  
  
  
  set.seed(42)

  folded_data_recipe_xgbmatrix <- folded_data_recipe_xgbmatrix %>% filter(fold %in% do_folds_step2)

  folded_data_recipe_xgbmatrix_xgbcv <-
    folded_data_recipe_xgbmatrix %>%
    mutate(
      xgcv = map(
        xgtrain,
        ~ xgb.cv(
          params = xgb_params,
          data = .x,
          nround = max_rounds,
          nfold = n_folds_step2,
          showsd = TRUE,
          early_stopping_round = 50
        )
      ),
      xgcv_best_test_rmse_mean = map_dbl(xgcv, ~ .x$evaluation_log$test_rmse_mean[.x$best_iteration]),
      xgcv_best_iteration = map_dbl(xgcv, ~ .x$best_iteration),
      xgmodel = map2(
        xgtrain, xgcv_best_iteration,
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
      truth = map(test, ~ .x %>% pull(claim_amount)),
      estimate = map2(xgtest, xgmodel, ~ predict(.y, newdata = .x)),
      test_rmse = map2_dbl(truth, estimate, ~ rmse_vec(.x, .y)),
      test_gini = map2_dbl(truth, estimate, ~ NormalizedGini(.y, .x))
    )



  test_w_preds <- bind_rows(folded_data_recipe_xgbmatrix_xgbcv_metrics$test) %>%
    bind_cols(
      tibble(
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
      params = list(xgb_params),
      estimate = list(test_w_preds$estimate),
      test_w_preds = list(test_w_preds)
    )
  write_rds(model_metrics, paste0("output/", model_name, submodel_start_time, "_model_metrics.rds"))
  write_csv(model_metrics %>% select(model_name, test_rmse, test_gini, mean_xgcv_best_iteration), paste0("output/", model_name, submodel_start_time, "_model_metrics.csv"))
  # WE have gone through all this to estimate out of fold performance using our whole data set as a "test" dataset.

  purrr::map2(folded_data_recipe_xgbmatrix_xgbcv$fold, folded_data_recipe_xgbmatrix_xgbcv$xgmodel,
              function(x,y){ xgb.save(y, fname = paste0("output/", model_name, submodel_start_time, "_fold",x,"_xgmodel.xgb"))})
  
  return(model_metrics)
}
