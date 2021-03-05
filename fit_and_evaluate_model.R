fit_and_evaluate_model <- function(folded_data_recipe_xgbmatrix,
                      model_name,
                     xgb_params
                      ) {
  set.seed(42)
  
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
      params = list(xgb_params),
      estimate = list(test_w_preds$estimate)
      
    )
  write_rds(model_metrics, paste0("output/",model_name, "_model_metrics.rds"))
  write_csv(model_metrics %>% select(model_name, test_rmse, test_gini, mean_xgcv_best_iteration), paste0("output/",model_name, "_model_metrics.csv"))
  # WE have gone through all this to estimate out of fold performance using our whole data set as a "test" dataset.
  
 return(model_metrics) 
}
