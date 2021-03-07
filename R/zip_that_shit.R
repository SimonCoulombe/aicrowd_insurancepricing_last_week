zip_everything <- function(zipname= "default.zip"){
zip::zip(
  zipfile = zipname,
  files = c("config.json",
            "R",
            "fit_model.R",
            "install.R",
            "predict.R",
            "training_data.csv",
            "prod"
            )
)
}