preprocess_X_data <- function(x_raw) {
  # Data preprocessing function: given X_raw, clean the data for training or prediction.

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.

  # Returns
  # -------
  # A cleaned / preprocessed version of the dataset

  # YOUR CODE HERE ------------------------------------------------------
  recipe1 <- read_rds("prod/prepped_first_recipe.rds")
  df1 <- bake(recipe1, new_data = x_raw) %>%
    bind_cols(x_raw %>% select(applied_car_ratio, applied_town_id_ratio, ajust_n_claim, ajust_age))




  # ---------------------------------------------------------------------
  # The result trained_model is something that you will save in the next section
  return(df1) # change this to return the cleaned data
}
