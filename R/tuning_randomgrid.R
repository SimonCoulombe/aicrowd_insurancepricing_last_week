
## OK I did something really overkill with B- randomgrid and C- bayesian.
# bercause I felt I didnt have much data, I turnede my whole dataset into a test set.
# steps:
# create 5 20% folds.
# do xgb.cv on fold A B C  D to find optimal number of iterations
# train model on A B C D with optimal number of iterations
# predict on fold E.  (the holdout folds)

# repeat  4 more times, using the different folds as holdout.
# don't think it was worth it.

tuning_randomgrid <- function() {
  
  # We are going to set up a system that allows us to use 100% of the training set for training and testing.
  # This is important because we don'T have that much data, so keeping only 20% of that is likely to be volatile.

  # assign all policies to a fold
  folds <- create__group_folds(training, id_policy)

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
      trained_recipe = map(train, ~ train_my_recipe_xgb_kitchensink(.x)),
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
  
  # # I love these parameters and I always want to have them around.
  simon_params <- data.frame(
    # booster = "gbtree",
    # objective = "reg:tweedie",
    # eval_metric = "rmse",
    # tweedie_variance_power = 1.66,
    gamma = 10,
    max_depth = 3,
    eta = 0.05,
    min_child_weight = 25,
    subsample = 0.6,
    colsample_bytree = 0.3,
    nrounds = 2000
    # tree_method = tree_method
  ) %>%
    as_tibble()


  # generate 20 random models parameter sets
  how_many_models <- 20
  set.seed(42)
  gamma <- data.frame(gamma = c(rep(0, how_many_models / 4), runif(3 * how_many_models / 4) * 10)) # 0 à 20
  eta <- data.frame(eta = c(rep(0.05, how_many_models))) # 0 à 10
  nrounds <- data.frame(nrounds = c(rep(2000, how_many_models))) # 0 à 10
  max_depth <- data.frame(max_depth = floor(runif(how_many_models) * 11) + 3) # 1 à 10
  min_child_weight <- data.frame(min_child_weight = floor(runif(how_many_models) * 100) + 1) # 1 à 100
  subsample <- data.frame(subsample = runif(how_many_models) * 0.8 + 0.2) # 0.2 à 1
  colsample_bytree <- data.frame(colsample_bytree = runif(how_many_models) * 0.8 + 0.2) # 0.2 à 1
  random_grid <- gamma %>%
    bind_cols(eta) %>%
    bind_cols(nrounds) %>%
    bind_cols(max_depth) %>%
    bind_cols(min_child_weight) %>%
    bind_cols(subsample) %>%
    bind_cols(colsample_bytree) %>%
    as_tibble()
  # combine random and hardcoded parameters
  df.params <- simon_params %>%
    bind_rows(random_grid) %>%
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


  # the get_random_grid_results tells us which parameter set has the best out-of-fold performance
  get_random_grid_results <- function(list_of_param_sets) {
    random_grid_results <- list_of_param_sets %>%
      mutate(booster = map2(data, xgb_params, function(X, Y) {
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
          folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
          model_name = paste0(X$rownumber),
          xgb_params = Y,
          do_folds = c(1)
        )

        message(paste0("Model_name =", paste0(X$rownumber), " test_gini: ", model_metrics$test_gini, " test_rmse", model_metrics$test_rmse))
        return(model_metrics)
      }))
  }


  tictoc::tic() # 34s
  random_grid_results <- get_random_grid_results(list_of_param_sets)
  random_grid_results <- random_grid_results %>% mutate(test_rmse = map_dbl(booster, ~ .x$test_rmse), test_gini = map_dbl(booster, ~ .x$test_gini))
  tictoc::toc()
  write_rds(random_grid_results, paste0("output/", start_time, "_random_grid_results.rds"))


  # est-ce que je peux retrouver le gini avec les estimates? yes!
  # test_pour_gini <- bind_rows(folded_data$test)  %>% bind_cols(tibble(estimate = test$booster[[1]]$estimate[[1]]))
  # MLmetrics::NormalizedGini(test_pour_gini$estimate, test_pour_gini$claim_amount)
  # 0.3372843

  best_xgb_params <- random_grid_results %>%
    arrange(-test_gini) %>%
    head(1) %>%
    pull(xgb_params) %>%
    .[[1]]
  return(best_xgb_params)
}
