fit_model <- function(x_raw= NULL, 
                      y_raw = NULL,
                      folds = NULL) {
  
  random_rounds <- 25
  bayesian_rounds <-25
  max_xgbcv_rounds <- 2000
  start_time <- format(Sys.time(), "%Y_%m_%d_%Hh%Mm%Ss")
  # Model training function: given training data (X_raw, y_raw), train this pricing model.
  
  # Parameters
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.
  # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
  # 	A one dimensional array, with values either 0 (most entries) or >0.
  
  # Returns
  # self: (optional), this instance of the fitted model.
  
  tree_method = "hist" #"gpu_hist" #"hist"
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
  training <- training %>% 
    mutate(claim_amount_backup = claim_amount,
           claim_amount = pmin(claim_amount, 10000)) # cap à 10 000$
  # We are going to set up a system that allows us to use 100% of the training set for training and testing.  
  # This is important because we don'T have that much data, so keeping only 20% of that is likely to be volatile.  
  
  
  ### create 5 folds  ----
  if(is.null(folds)){
    # set.seed(42)
    # policynumbers <-  training %>% distinct(id_policy) # everyone has 4 records, always put someone's records in the same fold
    # folds <- policynumbers[sample(nrow(policynumbers)),] %>%  # randomize policynumber order, then assign  folds according to row number
    #   mutate(fold = row_number() %% 5 +1) 
    # write_csv(folds, "folds/folds.csv")
    folds <- read_csv("folds/folds.csv")
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
        light_slow = if_else(vh_weight < 400 & vh_speed<130, 1, 0, NA_real_),
        light_fast = if_else(vh_weight < 400 & vh_speed>200, 1, 0, NA_real_),
        town_id = paste(population, 10*town_surface_area, sep = "_"),
        age_when_licensed =  drv_age1  - drv_age_lic1 ,
        pop_density = population / town_surface_area,
        young_man_drv1 = as.integer((drv_age1 <=24 & drv_sex1 == "M")),
        fast_young_man_drv1 = as.integer((drv_age1 <=30 & drv_sex1 == "M" & vh_speed >=200)),
        young_man_drv2 = as.integer((drv_age2 <=24 & drv_sex2 == "M")),
        #no_known_claim_values = as.integer(pol_no_claims_discount %in% no_known_claim_values),
        year = if_else(year <= 4, year, 4),  # replace year 5 with a 4.
        #year = as_factor(year),
        vh_current_value = vh_value * 0.8^(vh_age -1),  #depreciate 20% per year
        vh_time_left = pmax(20 - vh_age,0),
        pol_coverage_int = case_when(
          pol_coverage == "Min" ~ 1,
          pol_coverage == "Med1" ~ 2,
          pol_coverage == "Med2" ~ 3,
          pol_coverage == "Max" ~4),
        pol_pay_freq_int = case_when(
          pol_pay_freq == "Monthly" ~ 1,
          pol_pay_freq == "Quarterly" ~ 2,
          pol_pay_freq == "Biannual" ~ 3,
          pol_pay_freq == "Yearly" ~ 4
        )
      ) %>%
      recipes::step_string2factor(recipes::all_nominal()) %>%
      # 2 way interact
      recipes::step_interact(~ pol_coverage_int:vh_current_value) %>%
      recipes::step_interact(~ pol_coverage_int:vh_time_left) %>%
      recipes::step_interact(~ pol_coverage_int:pol_no_claims_discount) %>%
      recipes::step_interact(~ vh_current_value:vh_time_left) %>%  
      recipes::step_interact(~ vh_current_value:pol_no_claims_discount) %>%
      recipes::step_interact(~ vh_time_left:pol_no_claims_discount) %>%
      # 3 way intertac 
      recipes::step_interact(~ pol_coverage_int:vh_current_value:vh_age) %>%
      # remove id
      step_rm(contains("id_policy")) %>% 
      recipes::step_other(recipes::all_nominal(), threshold = 0.005) %>%
      recipes::step_dummy(all_nominal(), one_hot = TRUE)
    
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
      feature_names =  map(baked_train, ~ .x %>% select(-claim_amount, -year, -claim_amount_backup) %>% colnames()) # drop year
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
  
  # A - get model metrics manually ----
  
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
  
  # B - find the most promising model with  random grid  ----
  # 
  # # I love these parameters and I always want to have them around.  
  # simon_params <- data.frame(
  #   #booster = "gbtree",
  #   #objective = "reg:tweedie",
  #   #eval_metric = "rmse",
  #   #tweedie_variance_power = 1.66,
  #   gamma = 10,
  #   max_depth = 3,
  #   eta = 0.03,
  #   min_child_weight = 25,
  #   subsample = 0.6,
  #   colsample_bytree = 0.3,
  #   nrounds = 3000
  #   #tree_method = tree_method
  # ) %>% 
  #   as_tibble()
  # 
  # 
  # # generate 20 random models parameter sets
  # how_many_models <- 4
  # set.seed(42)
  # gamma <-            data.frame(gamma =c(rep(0,how_many_models/4), runif(3*how_many_models/4)*10)) # 0 à 20
  # eta <-              data.frame(eta =c(rep(0.1,how_many_models))) # 0 à 10
  # nrounds <-              data.frame(nrounds =c(rep(3000,how_many_models))) # 0 à 10
  # max_depth <-        data.frame(max_depth = floor(runif(how_many_models)*11 ) + 3)  # 1 à 10
  # min_child_weight <- data.frame(min_child_weight = floor(runif(how_many_models) * 100) + 1) # 1 à 100
  # subsample <-        data.frame(subsample =runif(how_many_models) * 0.8 + 0.2) # 0.2 à 1
  # colsample_bytree <- data.frame(colsample_bytree =runif(how_many_models) * 0.8 + 0.2)  # 0.2 à 1
  # random_grid <-gamma %>%
  #   bind_cols(eta ) %>%
  #   bind_cols(nrounds ) %>%
  #   bind_cols(max_depth ) %>%
  #   bind_cols(min_child_weight) %>%
  #   bind_cols(subsample) %>%
  #   bind_cols(colsample_bytree )   %>%
  #   as_tibble()
  # # combine random and hardcoded parameters
  # df.params <- simon_params %>%  bind_rows(random_grid) %>%
  #   mutate(rownum = row_number(),
  #          rownumber = row_number())
  # list_of_param_sets <- df.params %>% 
  #   group_nest(rownum)  %>% 
  #   mutate(xgb_params = map(data, 
  #                           ~  list(
  #                             booster = "gbtree",
  #                             eta = .x$eta,
  #                             max_depth = .x$max_depth,
  #                             min_child_weight = .x$min_child_weight,
  #                             gamma = .x$gamma,
  #                             subsample = .x$subsample,
  #                             colsample_bytree = .x$colsample_bytree,
  #                             objective = 'reg:tweedie', 
  #                             eval_metric = "rmse",
  #                             tweedie_variance_power = 1.66,
  #                             tree_method = tree_method
  #                           )
  #   )
  #   )
  # 
  # 
  # # the get_random_grid_results tells us which parameter set has the best out-of-fold performance 
  # get_random_grid_results <- function(list_of_param_sets){
  #   
  #   random_grid_results <- list_of_param_sets %>% 
  #     mutate(booster = map2(data, xgb_params, function(X, Y){
  #       message(paste0("model #",       X$rownumber,
  #                      " eta = ",              X$eta,
  #                      " max.depth = ",        X$max_depth,
  #                      " min_child_weigth = ", X$min_child_weight,
  #                      " subsample = ",        X$subsample,
  #                      " colsample_bytree = ", X$colsample_bytree,
  #                      " gamma = ",            X$gamma,
  #                      " nrounds = ",            X$nrounds
  #       )
  #       )
  # 
  #       model_metrics <- fit_and_evaluate_model(
  #         folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
  #         model_name = paste0(X$rownumber),
  #         xgb_params = Y
  #       )
  #       
  # 
  #       
  #       message(paste0("Model_name =", paste0(X$rownumber), " test_gini: ", model_metrics$test_gini, " test_rmse", model_metrics$test_rmse ))
  #       return(model_metrics)})
  #       )
  #     
  #   
  # }
  # 
  # 
  # tictoc::tic() #34s
  # random_grid_results <- get_random_grid_results(list_of_param_sets)
  # random_grid_results <- random_grid_results %>% mutate(test_rmse = map_dbl(booster, ~ .x$test_rmse), test_gini = map_dbl(booster, ~ .x$test_gini))
  # tictoc::toc()
  # write_rds(random_grid_results, paste0("output/", start_time, "_random_grid_results.rds"))
  # 
  # 
  # # est-ce que je peux retrouver le gini avec les estimates? yes!
  # #test_pour_gini <- bind_rows(folded_data$test)  %>% bind_cols(tibble(estimate = test$booster[[1]]$estimate[[1]])) 
  # #MLmetrics::NormalizedGini(test_pour_gini$estimate, test_pour_gini$claim_amount)
  # #0.3372843
  # 
  # best_xgb_params <- random_grid_results %>% arrange(-test_gini) %>% head(1) %>% pull(xgb_params) %>% .[[1]]
  # 
  # C - find the most promising parameters with bayesian search  -----
  
  # objective function: we want to maximise the log likelihood by tuning most parameters
  obj.fun  <- smoof::makeSingleObjectiveFunction(
    name = "xgb_cv_bayes",
    fn =   function(x){
      
      xgb_params =  list(
        booster          = "gbtree",
        #eta              = x["eta"],
        eta = 0.03, 
        max_depth        = x["max_depth"],
        min_child_weight = x["min_child_weight"],
        gamma            = x["gamma"],
        subsample        = x["subsample"],
        colsample_bytree = x["colsample_bytree"],
        objective = 'reg:tweedie', 
        eval_metric = "rmse",
        tree_method = tree_method
      )
      
      model_metrics <- fit_and_evaluate_model(
                 folded_data_recipe_xgbmatrix = folded_data_recipe_xgbmatrix,
                 model_name = "default",
                 xgb_params = xgb_params,
                 max_rounds= max_xgbcv_rounds
               )
      
      return(model_metrics$test_gini)
    },
    par.set = makeParamSet(
      makeNumericParam("gamma",            lower = 0,     upper = 20),
      makeIntegerParam("max_depth",        lower= 1,      upper = 8),
      makeIntegerParam("min_child_weight", lower= 1,      upper = 100),
      makeNumericParam("subsample",        lower = 0.1,   upper = 1),
      makeNumericParam("colsample_bytree", lower = 0.1,   upper = 1)
    ),
    minimize = FALSE # on veut minimiser gini!
  )
  
  # generate an optimal design with only 40  points
  
  
  set.seed(42)
  des = generateDesign(n= random_rounds,
                       par.set = getParamSet(obj.fun), 
                       fun = lhs::randomLHS) 
  
  
  simon_params <- data.frame(
    gamma = 10,
    max_depth = 3,
    # eta = 0.03,
    min_child_weight = 25,
    subsample = 0.6,
    colsample_bytree = 0.3
  ) %>%
    as_tibble()
  
  final_design = simon_params %>%
    bind_rows(des)
  #final_design = des
  # bayes will have 40 additional iterations
  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = bayesian_rounds)
  
  run = mbo(fun = obj.fun, 
            design = final_design,  
            control = control, 
            show.info = TRUE)
  write_rds( run, "output/run.rds")
  
  run$opt.path$env$path  %>% 
    mutate(Round = row_number()) %>%
    mutate(type = case_when(
      Round==1  ~ "1- hardcoded",
      Round<= random_rounds + 1 ~ "2 -random rounds",
      TRUE ~ "3 - bayesian rounds")) %>%
    ggplot(aes(x= Round, y= y, color= type)) + 
    geom_point() +
    labs(title = "mlrMBO optimization")+
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
  
  bayesian_params <-  run$opt.path$env$path %>% arrange(desc(y)) %>% head(1)# on veut le plus gros gini!
  write_csv(bayesian_params, "output/bayesian_params.csv")
  
  best_xgb_params <- list(
    booster = "gbtree",
    objective = "reg:tweedie",
    eval_metric = "rmse",
    tweedie_variance_power = 1.66,
    gamma = bayesian_params$gamma,
    max_depth = bayesian_params$max_depth,
    eta = 0.01, #best_params$eta, # learn slower for this model
    min_child_weight = bayesian_params$min_child_weight,
    subsample = bayesian_params$subsample,
    colsample_bytree = bayesian_params$colsample_bytree,
    tree_method = tree_method
  )
  
  
  ## ok we decide that we like these xgb_params best.  Let's fit the most promising model once and for all
  # ----fit most promising model ----
  
  
  
  
  
  trained_recipe <- train_my_recipe(training)
  baked_training <- bake(trained_recipe, new_data = training)
  feature_names =  baked_training %>% select(-claim_amount, -year, -claim_amount_backup) %>% colnames()
  xgtrain <-    xgb.DMatrix(
    as.matrix(baked_training %>% select(all_of(feature_names))),
    label = baked_training %>% pull(claim_amount)
  )
  set.seed(42)
  xgcv <- xgb.cv(
    params = best_xgb_params,
    data = xgtrain,
    nround = 4000,
    nfold = 5,
    showsd = TRUE,
    early_stopping_round = 100
  )
  
  xgcv_best_test_rmse_mean = xgcv$evaluation_log$test_rmse_mean[xgcv$best_iteration]
  xgcv_best_iteration =  xgcv$best_iteration
  set.seed(42)
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
  claim_amount_backup <- training$claim_amount_backup
  
  ajusts <- tibble(ajusts = sum(claim_amount_backup) / sum(preds))
  write_csv(ajusts,paste0("output/", start_time, "_ajusts.csv"))
  
  return(list(xgmodel = xgmodel, ajusts = ajusts, trained_recipe = trained_recipe)) # return(trained_model)
  
  
}

