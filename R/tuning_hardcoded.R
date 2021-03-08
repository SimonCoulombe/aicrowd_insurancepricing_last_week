tuning_hardcoded <- function(){
  
simon_params1 <- data.frame(
  gamma = 10,
  max_depth = 3,
  eta = 0.05,
  min_child_weight = 25,
  subsample = 0.6,
  colsample_bytree = 0.3,
  nrounds = 2000
) %>%
  as_tibble()

simon_params2 <-  list(
  gamma = 4,
  max_depth = 4,
  eta = 0.05,
  min_child_weight = 10,
  subsample = 0.6,
  colsample_bytree = 0.4,
  nrounds = 2000
) %>%
  as_tibble()

df.params <- bind_rows(simon_params1, simon_params2) %>%
  mutate(
    rownum = row_number(),
    rownumber = row_number()
  )
list_of_param_sets <- df.params %>%
  group_nest(rownum) %>%
  mutate(xgb_params = map(
    data,
    ~ list(
      booster = "gbtree",
      eta = .x$eta,
      max_depth = .x$max_depth,
      min_child_weight = .x$min_child_weight,
      gamma = .x$gamma,
      subsample = .x$subsample,
      colsample_bytree = .x$colsample_bytree,
      objective = "reg:tweedie",
      eval_metric = "rmse",
      tweedie_variance_power = 1.66,
      tree_method = tree_method
    )
  ))


z_kitchensink <- evaluate_multiple_parameters(
  list_of_param_sets,
  do_folds_step2 = c(3),
  max_rounds = 2000,
  recipe_function = train_my_recipe_xgb_kitchensink
)
# z_kitchensink
# > z_kitchensink
# # A tibble: 2 x 14
# rownum           data xgb_params        model_name test_rmse test_gini mean_xgcv_best_iteration xgcv_best_test_rmse_means xgcv_best_iterations test_rmses test_ginis params            estimate       test_w_preds
# <int> <list<tibble>> <list>            <chr>          <dbl>     <dbl>                    <dbl> <list>                    <list>               <list>     <list>     <list>            <list>         <list>
#   1      1        [1 × 8] <named list [11]> 1               501.     0.336                      240 <dbl [1]>                 <dbl [1]>            <dbl [1]>  <dbl [1]>  <named list [11]> <dbl [45,644]> <spec_tbl_df [45,644 × 30]>
#   2      2        [1 × 8] <named list [11]> 2               501.     0.330                      226 <dbl [1]>                 <dbl [1]>            <dbl [1]>  <dbl [1]>  <named list [11]> <dbl [45,644]> <spec_tbl_df [45,644 × 30]>

z_no_step_interact <- evaluate_multiple_parameters(
  list_of_param_sets,
  do_folds_step2 = c(3),
  max_rounds = 2000,
  recipe_function = train_my_recipe_xgb_no_step_interact
)
# z_no_step_interact
# # A tibble: 2 x 14
# rownum           data xgb_params        model_name test_rmse test_gini mean_xgcv_best_iteration xgcv_best_test_rmse_means xgcv_best_iterations test_rmses test_ginis params            estimate       test_w_preds
# <int> <list<tibble>> <list>            <chr>          <dbl>     <dbl>                    <dbl> <list>                    <list>               <list>     <list>     <list>            <list>         <list>
#   1      1        [1 × 8] <named list [11]> 1               501.     0.334                      301 <dbl [1]>                 <dbl [1]>            <dbl [1]>  <dbl [1]>  <named list [11]> <dbl [45,644]> <spec_tbl_df [45,644 × 30]>
#   2      2        [1 × 8] <named list [11]> 2               501.     0.333                      186 <dbl [1]>                 <dbl [1]>            <dbl [1]>  <dbl [1]>  <named list [11]> <dbl [45,644]> <spec_tbl_df [45,644 × 30]>

# # turns out the best model according to gini is the first set of hyperparameters with the  kitchensink recipe.. let's do this! but let'S also divide eta by 10
best_xgb_params  <- bind_rows(z_kitchensink, z_no_step_interact ) %>%
  arrange(desc(test_gini)) %>%
  pull(xgb_params) %>%
  .[[1]]

return(best_xgb_params )
}