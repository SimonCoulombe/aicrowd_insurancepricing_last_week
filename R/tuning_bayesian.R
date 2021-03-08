tuning_bayesian <- function() {
  
  
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
  
  
  obj.fun <- smoof::makeSingleObjectiveFunction(
    name = "xgb_cv_bayes",
    fn = function(x) {
      xgb_params <- list(
        booster = "gbtree",
        # eta              = x["eta"],
        eta = 0.03,
        max_depth = x["max_depth"],
        min_child_weight = x["min_child_weight"],
        gamma = x["gamma"],
        subsample = x["subsample"],
        colsample_bytree = x["colsample_bytree"],
        objective = "reg:tweedie",
        eval_metric = "rmse",
        tree_method = tree_method
      )

      model_metrics <- fit_and_evaluate_model(
        folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
        model_name = "default",
        xgb_params = xgb_params,
        max_rounds = max_xgbcv_rounds,
        do_folds = c(1, 2)
      )

      return(model_metrics$test_gini)
    },
    par.set = makeParamSet(
      makeNumericParam("gamma", lower = 0, upper = 20),
      makeIntegerParam("max_depth", lower = 1, upper = 6),
      makeIntegerParam("min_child_weight", lower = 1, upper = 100),
      makeNumericParam("subsample", lower = 0.1, upper = 0.8),
      makeNumericParam("colsample_bytree", lower = 0.1, upper = 0.8)
    ),
    minimize = FALSE # on veut minimiser gini!
  )

  # generate an optimal design with only 40  points


  set.seed(42)
  des <- generateDesign(
    n = random_rounds,
    par.set = getParamSet(obj.fun),
    fun = lhs::randomLHS
  )


  simon_params <- data.frame(
    gamma = 10,
    max_depth = 3,
    # eta = 0.03,
    min_child_weight = 25,
    subsample = 0.6,
    colsample_bytree = 0.3
  ) %>%
    as_tibble()

  final_design <- simon_params %>%
    bind_rows(des)
  # final_design = des
  # bayes will have 40 additional iterations
  control <- makeMBOControl()
  control <- setMBOControlTermination(control, iters = bayesian_rounds)

  run <- mbo(
    fun = obj.fun,
    design = final_design,
    control = control,
    show.info = TRUE
  )
  write_rds(run, paste0("output/", start_time, "_run.rds"))
  write_csv(run$opt.path$env$path, paste0("output/", start_time, "_bayesian_run.csv"))
  run$opt.path$env$path %>%
    mutate(Round = row_number()) %>%
    mutate(type = case_when(
      Round == 1 ~ "1- hardcoded",
      Round <= random_rounds + 1 ~ "2 -random rounds",
      TRUE ~ "3 - bayesian rounds"
    )) %>%
    ggplot(aes(x = Round, y = y, color = type)) +
    geom_point() +
    labs(title = "mlrMBO optimization") +
    ylab("gini")

  # max_depth =6 sinon ça pète avec gpu_hist https://discuss.xgboost.ai/t/when-using-xgb-cv-with-gpu-hist-check-failed-slice-only-supported-for-simpledmatrix-currently/1583/4

  # gain de 0.0396 entre mes paramètres et les meilleures..
  # > run$opt.path$env$path
  # gamma max_depth min_child_weight subsample colsample_bytree        y
  # 1   10.000000000         3               25 0.6000000        0.3000000 711.9276

  # > run$opt.path$env$path %>%  filter(y == min(y))
  # gamma max_depth min_child_weight subsample colsample_bytree       y
  # 1 15.5261         3               31 0.6317372         0.100022 711.888


  # ok on va devoir retourner chercher le nombre d'arbres maintenant.

  bayesian_params <- run$opt.path$env$path %>%
    arrange(desc(y)) %>%
    head(1) # on veut le plus gros gini!
  write_csv(bayesian_params, paste0("output/", start_time, "_bayesian_params.csv"))

  best_xgb_params <- list(
    booster = "gbtree",
    objective = "reg:tweedie",
    eval_metric = "rmse",
    tweedie_variance_power = 1.66,
    gamma = bayesian_params$gamma,
    max_depth = bayesian_params$max_depth,
    eta = 0.01, # best_params$eta, # learn slower for this model
    min_child_weight = bayesian_params$min_child_weight,
    subsample = bayesian_params$subsample,
    colsample_bytree = bayesian_params$colsample_bytree,
    tree_method = tree_method
  )

  return(best_xgb_params)
}
