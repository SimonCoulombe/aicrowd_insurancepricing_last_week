save_model <- function(model, output_path) {
  # Saves this trained model to a file.

  # This is used to save the model after training, so that it can be used for prediction later.

  # Do not touch this unless necessary (if you need specific features). If you do, do not
  #  forget to update the load_model method to be compatible.

  # Save in `trained_model.RData`.

  save(model, file = output_path)
}
