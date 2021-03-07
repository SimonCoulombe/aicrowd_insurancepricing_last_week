# load libraries and functions -----
library(tidyverse)
library(tidymodels)
library(xgboost)
library(MLmetrics)
library(mlrMBO) # for bayesian optimisation
require("DiceKriging") # mlrmbo requires this
require("rgenoud") # mlrmbo requires this

purrr::walk(
  list.files("R", full.names = TRUE, pattern = "*.R"),
  source
)

# set some parameters and switchs
random_rounds <- 5
bayesian_rounds <- 5
max_xgbcv_rounds <- 2000
tree_method <- "hist" # "gpu_hist

# define new functions ----
create__group_folds <- function(df, group_var = id_policy, n_folds = 5, seed = 42) {
  set.seed(seed)
  group_var_values <- df %>% 
    distinct({{ group_var }}) # everyone has 4 records, always put someone's records in the same fold
  folds <- group_var_values[sample(nrow(group_var_values)), ] %>% # randomize policynumber order, then assign  folds according to row number
    mutate(fold = row_number() %% n_folds + 1)
}

train_my_recipe <- function(.data) {
  my_first_recipe <-
    recipes::recipe(
      claim_amount ~ .,
      .data[0, ]
    ) %>%
    recipes::step_mutate(
      light_slow = if_else(vh_weight < 400 & vh_speed < 130, 1, 0, NA_real_),
      light_fast = if_else(vh_weight < 400 & vh_speed > 200, 1, 0, NA_real_),
      town_id = paste(population, 10 * town_surface_area, sep = "_"),
      age_when_licensed = drv_age1 - drv_age_lic1,
      pop_density = population / town_surface_area,
      young_man_drv1 = as.integer((drv_age1 <= 24 & drv_sex1 == "M")),
      fast_young_man_drv1 = as.integer((drv_age1 <= 30 & drv_sex1 == "M" & vh_speed >= 200)),
      young_man_drv2 = as.integer((drv_age2 <= 24 & drv_sex2 == "M")),
      # no_known_claim_values = as.integer(pol_no_claims_discount %in% no_known_claim_values),
      year = if_else(year <= 4, year, 4), # replace year 5 with a 4.
      vh_current_value = vh_value * 0.8^(vh_age - 1), # depreciate 20% per year
      vh_time_left = pmax(20 - vh_age, 0),
      pol_coverage_int = case_when(
        pol_coverage == "Min" ~ 1,
        pol_coverage == "Med1" ~ 2,
        pol_coverage == "Med2" ~ 3,
        pol_coverage == "Max" ~ 4
      ),
      pol_pay_freq_int = case_when(
        pol_pay_freq == "Monthly" ~ 1,
        pol_pay_freq == "Quarterly" ~ 2,
        pol_pay_freq == "Biannual" ~ 3,
        pol_pay_freq == "Yearly" ~ 4
      )
    ) %>%
    recipes::step_other(recipes::all_nominal(), threshold = 0.005) %>%
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
    # recipes::step_novel(all_nominal()) %>%
    recipes::step_dummy(all_nominal(), one_hot = TRUE)
  prepped_first_recipe <- recipes::prep(my_first_recipe, .data, retain = FALSE)
  return(prepped_first_recipe)
}

add_equal_weight_group <- function(table, sort_by, expo = NULL, group_variable_name = "groupe", nb = 10) {
  sort_by_var <- enquo(sort_by)
  groupe_variable_name_var <- enquo(group_variable_name)
  if (!(missing(expo))) { # https://stackoverflow.com/questions/48504942/testing-a-function-that-uses-enquo-for-a-null-parameter
    expo_var <- enquo(expo)
    total <- table %>%
      pull(!!expo_var) %>%
      sum()
    br <- seq(0, total, length.out = nb + 1) %>%
      head(-1) %>%
      c(Inf) %>%
      unique()
    table %>%
      arrange(!!sort_by_var) %>%
      mutate(cumExpo = cumsum(!!expo_var)) %>%
      mutate(!!group_variable_name := cut(cumExpo, breaks = br, ordered_result = TRUE, include.lowest = TRUE) %>% as.numeric()) %>%
      select(-cumExpo)
  } else {
    total <- nrow(table)
    br <- seq(0, total, length.out = nb + 1) %>%
      head(-1) %>%
      c(Inf) %>%
      unique()
    table %>%
      arrange(!!sort_by_var) %>%
      mutate(cumExpo = row_number()) %>%
      mutate(!!group_variable_name := cut(cumExpo, breaks = br, ordered_result = TRUE, include.lowest = TRUE) %>% as.numeric()) %>%
      select(-cumExpo)
  }
}

#' fit_model train the insurance model(s)
#'
#' @param x_raw
#' @param y_raw
#' @param folds
#'
#' @return
#' @export
#'
#' @examples
fit_model <- function(x_raw = NULL,
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
  
  if (is.null(x_raw)) {
    training <- read_csv("training_data.csv")
  } else {
    training <- bind_cols(x_raw, y_raw)
  }
  training <- training %>%
    mutate(
      claim_amount_backup = claim_amount,
      claim_amount = pmin(claim_amount, 10000)
    ) # cap à 10 000$
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
      trained_recipe = map(train, ~ train_my_recipe(.x)),
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
  
  # set Hyperparameter -----
  # Either A) hard code, B) randomgrid or C) bayesian search
  # A - get model metrics manually
  #
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
  #
  #
  best_xgb_params <- list(
    booster = "gbtree",
    objective = "reg:tweedie",
    eval_metric = "rmse",
    tweedie_variance_power = 1.66,
    gamma = 4,
    max_depth = 4,
    eta = 0.01,
    min_child_weight = 10,
    subsample = 0.6,
    colsample_bytree = 0.4,
    tree_method = "hist"
  )
  # best_xgb_params <- tuning_randomgrid()
  # best_xgb_params <- tuning_bayesian()
  
  # ----fit most promising model ----
  trained_recipe <- train_my_recipe(training)
  
  baked_training <- bake(trained_recipe, new_data = training)
  
  feature_names <- baked_training %>%
    select(-claim_amount, -year, -claim_amount_backup) %>%
    colnames()
  
  xgtrain <- xgb.DMatrix(
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
    early_stopping_round = 100,
    prediction = TRUE
  )
  
  xgcv_best_test_rmse_mean <- xgcv$evaluation_log$test_rmse_mean[xgcv$best_iteration]
  
  xgcv_best_iteration <- xgcv$best_iteration
  
  final_model_oof_preds <- training %>%
    select(id_policy, year) %>%
    bind_cols(tibble(preds = xgcv$pred))
  
  final_model_metrics <- tibble(
    xgcv_best_test_rmse_mean,
    xgcv_best_iteration,
    rmse = rmse_vec(training$claim_amount, xgcv$pred),
    gini = NormalizedGini(xgcv$pred, training$claim_amount)
  )
  
  set.seed(42)
  
  xgmodel <- xgboost::xgb.train(
    data = xgtrain,
    params = best_xgb_params,
    nrounds = xgcv_best_iteration
  )
  
  my_importance <- xgb.importance(
    feature_names = feature_names,
    model = xgmodel
  ) %>%
    as_tibble()
  
  calibration_data <- training %>%
    left_join(final_model_oof_preds) %>%
    mutate(town_id = paste(population, 10 * town_surface_area, sep = "_"))
  #
  # # cool lift charts
  # p1 <-add_equal_weight_group(calibration_data, preds) %>%
  #   group_by(groupe) %>%
  #   summarise(preds =mean(preds), actual = mean(claim_amount)) %>%
  #   gather(key=key, value=value, -groupe) %>%
  #   ggplot(aes(x = groupe,y=value, color = key ))+ geom_point()
  #
  # p2 <-  calibration_data %>%
  #   left_join(n_claim_year1_to_year3) %>%
  #   filter(year == 4) %>%
  #   add_equal_weight_group(., preds) %>%
  #   group_by(groupe, n_claims) %>%
  #   summarise(preds =mean(preds), actual = mean(claim_amount)) %>%
  #   gather(key=key, value=value, -groupe, -n_claims) %>%
  #   ggplot(aes(x = groupe,y=value, color = key ))+ geom_point() + facet_wrap(~n_claims)
  #
  #
  # data_oof <- function(variable){
  #   calibration_data %>%
  #     group_by({{variable}}) %>%
  #     summarise(preds =mean(preds),
  #               actual = mean(claim_amount),
  #               n = n()
  #     ) %>%
  #     gather(key=key, value=value, -{{variable}}, -n)
  #
  # }
  #
  # plot_oof <- function(variable){
  #   data_oof({{variable}}) %>%
  #     ggplot(aes(x = {{variable}},y=value, color = key ))+ geom_point(aes(size=n), alpha =0.6) +
  #     theme_bw()
  # }
  #
  # data_oof2 <- function(variable){
  #   baked_training %>%
  #     bind_cols(final_model_oof_preds) %>%
  #     group_by({{variable}}) %>%
  #     summarise(preds =mean(preds),
  #               actual = mean(claim_amount),
  #               n = n()
  #     ) %>%
  #     gather(key=key, value=value, -{{variable}}, -n)
  # }
  # plot_oof2 <- function(variable){
  #   data_oof2({{variable}}) %>%
  #     ggplot(aes(x = {{variable}},y=value, color = key ))+ geom_point(aes(size=n), alpha =0.6)  +
  #     theme_bw()
  # }
  #
  # plot_oof(drv_age1)
  # plot_oof(drv_sex1)
  # plot_oof(pol_coverage)
  # plot_oof(vh_fuel)
  # plot_oof(vh_age)
  # plot_oof(vh_type)
  #
  # plot_oof2(age_when_licensed)
  # plot_oof2(drv_sex1_M)
  #
  # plot_oof_continuous <- function(variable){
  #   add_equal_weight_group(calibration_data, {{variable}} )%>%
  #     group_by(groupe) %>%
  #     summarise(preds =mean(preds),
  #               actual = mean(claim_amount),
  #               n = n()
  #     ) %>%
  #     gather(key=key, value=value, -groupe, -n) %>%
  #     ggplot(aes(x= groupe, y = value, color = key))+ geom_point(aes(size=n), alpha =0.6)  +
  #     theme_bw()
  #
  # }
  #
  #
  # plot_oof_continuous2 <- function(variable){
  #   add_equal_weight_group(baked_training %>%
  #                            bind_cols(final_model_oof_preds) , {{variable}} )%>%
  #     group_by(groupe) %>%
  #     summarise(preds =mean(preds),
  #               actual = mean(claim_amount),
  #               n = n()
  #     ) %>%
  #     gather(key=key, value=value, -groupe, -n) %>%
  #     ggplot(aes(x= groupe, y = value, color = key))+ geom_point(aes(size=n), alpha =0.6)  +
  #     theme_bw()
  #
  # }
  #
  #
  # plot_oof_continuous(vh_value)
  # plot_oof_continuous2(vh_current_value)
  #
  #
  
  # Create adjustments for variables that the model doesnt use (or doesnt use well, like age) ----
  # ajustment for age
  ajust_age <- calibration_data %>%
    group_by(drv_age1) %>%
    summarise(
      preds = sum(preds), actual = sum(claim_amount)) %>%
    mutate(ajust_age = if_else(
      drv_age1 < 25,  pmax(actual / preds, 1),
      1,
      1
    )) %>%
    arrange(drv_age1) %>%
    select(drv_age1, ajust_age)
  
  ajuste_pour_age <-
    calibration_data %>%
    left_join(ajust_age) %>%
    mutate(pred_age = preds * ajust_age)
  
  # ajustment for previous claims
  n_claim_year1_to_year3 <- training %>%
    filter(year <= 3) %>%
    group_by(id_policy) %>%
    summarise(n_claims = sum(claim_amount > 0))
  
  n_claim_year1_to_year4 <- training %>%
    filter(year <= 4) %>%
    group_by(id_policy) %>%
    summarise(n_claims = sum(claim_amount > 0))
  
  adjustments_for_n_claim <- ajuste_pour_age %>%
    left_join(n_claim_year1_to_year3) %>%
    filter(year == 4) %>%
    group_by(n_claims) %>%
    summarise(across(c(claim_amount, claim_amount_backup, preds, pred_age), sum)) %>%
    mutate(ajust_n_claim = pmax(claim_amount / pred_age, 1)) %>%
    bind_rows(tibble(n_claims = 4, ajust_n_claim = 10000))
  
  adjusted_for_clean_record <- ajuste_pour_age %>%
    left_join(n_claim_year1_to_year3) %>%
    filter(year == 4) %>%
    left_join(adjustments_for_n_claim %>% select(n_claims, ajust_n_claim)) %>%
    mutate(preds_n_claim = pred_age * ajust_n_claim)
  
  
  # ajust for risky car
  adjust_for_car <- adjusted_for_clean_record %>%
    group_by(vh_make_model) %>%
    summarise(preds = sum(preds_n_claim), actual = sum(claim_amount)) %>%
    mutate(
      ratio = actual / preds,
      applied_car_ratio = pmin(pmax(ratio, 1), 1.5)
    ) %>%
    arrange(desc(applied_car_ratio)) %>%
    select(vh_make_model, applied_car_ratio)
  
  adjusted_for_cars <- adjusted_for_clean_record %>%
    left_join(adjust_for_car %>% select(vh_make_model, applied_car_ratio)) %>%
    replace_na(list(applied_car_ratio = 1.5)) %>% # triple premium for unknown car
    mutate(preds_adjusted_for_car = preds_n_claim * applied_car_ratio)
  
  
  adjusted_for_cars %>%
    summarise(across(
      c(claim_amount, claim_amount_backup, preds, preds_n_claim, preds_adjusted_for_car),
      sum))
  # ajust for risky city
  adjust_for_city <- adjusted_for_cars %>%
    group_by(town_id) %>%
    summarise(preds = sum(preds_adjusted_for_car), actual = sum(claim_amount)) %>%
    mutate(
      ratio = actual / preds,
      applied_town_id_ratio = pmin(pmax(ratio, 1), 1.5)
    ) %>%
    arrange(desc(applied_town_id_ratio)) %>%
    select(town_id, applied_town_id_ratio)
  
  adjusted_for_cars_and_city <- adjusted_for_cars %>%
    left_join(adjust_for_city %>% select(town_id, applied_town_id_ratio)) %>%
    replace_na(list(applied_town_id_ratio = 1.5)) %>% # triple premium for unknown car
    mutate(preds_adjusted_for_car_town_id = preds_adjusted_for_car * applied_town_id_ratio)
  
  # save everything to output with unique name
  adjusted_for_cars_and_city %>%
    summarise(across(c(claim_amount, claim_amount_backup, preds, preds_n_claim, preds_adjusted_for_car, preds_adjusted_for_car_town_id), sum)) %>%
    write_csv(., paste0("output/final_model_", start_time, "_preds_after_ajust.csv"))
  
  write_rds(xgcv, paste0("output/final_model_", start_time, "xgcv.rds"))
  write_csv(n_claim_year1_to_year4, paste0("output/final_model_", start_time, "_n_claim_year1_to_year4.csv"))
  write_csv(adjustments_for_n_claim, paste0("output/final_model_", start_time, "_adjustments_for_n_claim.csv"))
  write_csv(ajust_age, paste0("output/final_model_", start_time, "_ajust_age.csv"))
  write_csv(adjust_for_car, paste0("output/final_model_", start_time, "_ajust_cars.csv"))
  write_csv(adjust_for_city, paste0("output/final_model_", start_time, "_ajust_city.csv"))
  write_csv(final_model_metrics, paste0("output/final_model_", start_time, "xgcv_metrics.csv"))
  write_csv(final_model_oof_preds, paste0("output/final_model_", start_time, "_oof_preds.csv"))
  write_csv(my_importance, paste0("output/final_model_", start_time, "_importance.csv"))
  write_csv(as_tibble(best_xgb_params), paste0("output/final_model_", start_time, "_xgb_params.csv"))
  write_rds(trained_recipe, paste0("output/final_model_", start_time, "_recipe.rds"))
  xgb.save(xgmodel, fname = paste0("output/final_model_", start_time, "_xgmodel.xgb"))
  ajusts <- tibble(ajusts = sum(adjusted_for_cars_and_city$claim_amount_backup) / sum(adjusted_for_cars_and_city$preds_adjusted_for_car_town_id))
  write_csv(ajusts, paste0("output/final_model_", start_time, "_ajusts.csv"))
  
  # save to prod with generic name
  xgb.save(xgmodel, fname = paste0("prod/trained_model.xgb"))
  write_rds(feature_names, "prod/feature_names.rds")
  write_csv(ajust_age, paste0("prod/ajust_age.csv"))
  write_csv(n_claim_year1_to_year4, "prod/n_claim_year1_to_year4.csv")
  write_csv(adjustments_for_n_claim, "prod/adjustments_for_n_claim.csv")
  write_csv(adjust_for_car, "prod/ajust_cars.csv")
  write_csv(adjust_for_city, "prod/ajust_city.csv")
  write_rds(trained_recipe, "prod/prepped_first_recipe.rds")
  
  # zip  for submission
  zip_everything(paste0(start_time, "_model.zip"))
  
    return(list(xgmodel = xgmodel, ajusts = ajusts, trained_recipe = trained_recipe)) # return(trained_model)
}
