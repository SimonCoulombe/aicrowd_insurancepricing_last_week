load_model <- function(model_path) {
  # Load a saved trained model from the file `trained_model.RData`.

  #    This is called by the server to evaluate your submission on hidden data.
  #    Only modify this *if* you modified save_model.

  model <- xgb.load(model_path)
  return(model)
}
