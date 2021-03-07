predict_expected_claim <- function(model, x_raw) {
  # Model prediction function: predicts the average claim based on the pricing model.

  # This functions estimates the expected claim made by a contract (typically, as the product
  # of the probability of having a claim multiplied by the average cost of a claim if it occurs),
  # for each contract in the dataset X_raw.

  # This is the function used in the RMSE leaderboard, and hence the output should be as close
  # as possible to the expected cost of a contract.

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.

  # Returns
  # -------
  # avg_claims: a one-dimensional array of the same length as X_raw, with one
  #     average claim per contract (in same order). These average claims must be POSITIVE (>0).

  xgmatrix <- xgb.DMatrix(
    as.matrix(x_raw %>% select(all_of(feature_names)))
  )

  y_predict <- predict(model, newdata = xgmatrix)

  return(y_predict)
}
