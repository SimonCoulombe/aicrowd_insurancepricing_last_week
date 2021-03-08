library(tidyverse)
library(tidymodels)
library(xgboost)

walk(
  list.files("R", full.names = TRUE, pattern = "*.R"),
  source
)
model_output_path <- "prod/trained_model.xgb"
feature_names <- read_rds("prod/feature_names.rds")
ajust_age <- read_csv("prod/ajust_age.csv")
ajust_cars <- read_csv("prod/ajust_cars.csv")
ajust_city <- read_csv("prod/ajust_city.csv")
ajust_n_claim <- read_csv("prod/adjustments_for_n_claim.csv")
n_claim_list <- read_csv("prod/n_claim_year1_to_year4.csv")

# This script expects sys.args arguments for (1) the dataset and (2) the output file.
output_dir <- Sys.getenv("OUTPUTS_DIR", ".")
input_dataset <- Sys.getenv("DATASET_PATH", "training_data.csv") # The default value.
output_claims_file <- paste(output_dir, "claims.csv", sep = "/") # The file where the expected claims should be saved.
output_prices_file <- paste(output_dir, "prices.csv", sep = "/") # The file where the prices should be saved.

if (!(interactive())) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) >= 1) {
    input_dataset <- args[1]
  }
  if (length(args) >= 2) {
    output_claims_file <- args[2]
  }
  if (length(args) >= 3) {
    output_prices_file <- args[3]
  }
} else {
  message("not interactive")
}

Xraw <- read_csv(input_dataset) %>%
  left_join(ajust_age) %>%
  left_join(n_claim_list) %>%
  left_join(ajust_n_claim %>% select(n_claims, ajust_n_claim)) %>%
  left_join(ajust_cars) %>%
  mutate(town_id = paste(population, 10 * town_surface_area, sep = "_")) %>%
  left_join(ajust_city) %>% # load the data
  replace_na(list(
    ajust_age = 1, 
    applied_car_ratio = 1.5, 
    applied_town_id_ratio = 1.5, 
    ajust_n_claim = 1.05)
  ) # unknown car or town = +50%.  no history = +5%

x_clean <- preprocess_X_data(Xraw) # clean the data
trained_model <- load_model(model_output_path) # load the model

if (Sys.getenv("WEEKLY_EVALUATION", "false") == "true") {
  prices <- predict_premium(trained_model, x_clean)
  write.table(x = prices, file = output_prices_file, row.names = FALSE, col.names = FALSE, sep = ",")
} else {
  claims <- predict_expected_claim(trained_model, x_clean)
  write.table(x = claims, file = output_claims_file, row.names = FALSE, col.names = FALSE, sep = ",")
}