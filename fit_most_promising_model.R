fit_most_promising_model <- function() {
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