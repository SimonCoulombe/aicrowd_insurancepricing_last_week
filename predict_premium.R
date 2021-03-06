source("predict_expected_claim.R")
source("preprocess_X_data.R")
predict_premium <- function(model, x_raw) {
  # Model prediction function: predicts premiums based on the pricing model.

  # This function outputs the prices that will be offered to the contracts in X_raw.
  # premium will typically depend on the average claim predicted in
  # predict_expected_claim, and will add some pricing strategy on top.

  # This is the function used in the average profit leaderboard. Prices output here will
  # be used in competition with other models, so feel free to use a pricing strategy.

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.

  # Returns
  # -------
  # prices: a one-dimensional array of the same length as X_raw, with one
  #     price per contract (in same order). These prices must be POSITIVE (>0).
  predicted_claims <- predict_expected_claim(model, x_raw)
  premiums <- predicted_claims * x_raw$ajust_n_claim * 1.19 * x_raw$applied_car_ratio  * x_raw$applied_town_id_ratio * x_raw$ajust_age # 1.19
  
  
  return(premiums)
}
