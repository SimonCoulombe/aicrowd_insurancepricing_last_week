library(tidyverse)
library(tidymodels)
library(xgboost)

#source("prep_recipe.R") # train wrangling recipes, only do onces
#source("fit_model.R")  # train xgboost, only do once.
#fit_model()
source("load_model.R")
source("predict_expected_claim.R")
source("predict_premium.R")
source("preprocess_X_data.R")

model_output_path <- "prod/trained_model.xgb"
feature_names <- read_rds("prod/feature_names.rds")
ajust_cars <- read_csv("prod/ajust_cars.csv")
ajust_city <- read_csv("prod/ajust_city.csv")
ajust_n_claim <- read_csv("prod/adjustments_for_n_claim.csv")
n_claim_list <- read_csv("prod/n_claim_year1_to_year4.csv")


# This script expects sys.args arguments for (1) the dataset and (2) the output file.
output_dir = Sys.getenv('OUTPUTS_DIR', '.')
input_dataset = Sys.getenv('DATASET_PATH', 'training_data.csv')  # The default value.
output_claims_file = paste(output_dir, 'claims.csv', sep = '/')  # The file where the expected claims should be saved.
output_prices_file = paste(output_dir, 'prices.csv', sep = '/')  # The file where the prices should be saved.


if(!(interactive())){
  args = commandArgs(trailingOnly=TRUE)
  
  if(length(args) >= 1){
    input_dataset = args[1]
  }
  if(length(args) >= 2){
    output_claims_file = args[2]
  }
  if(length(args) >= 3){
    output_prices_file = args[3]
  }
} else message("not interactive")

Xraw <- read_csv(input_dataset)  %>% 
   select(-claim_amount) %>% 
  #mutate(year=5)  %>% 
  #mutate(vh_make_model = "prout") %>%   #, population = 12, town_surface_area = 13) %>% 
  left_join(n_claim_list) %>% 
  left_join(ajust_n_claim) %>% 
  left_join(ajust_cars) %>%
  mutate(town_id = paste(population, 10*town_surface_area, sep = "_")) %>% 
  left_join(ajust_city) %>% # load the data
  replace_na(list(applied_car_ratio = 1.5, applied_town_id_ratio = 1.5, ajust_n_claim = 1.25))

x_clean <- preprocess_X_data(Xraw) # clean the data
trained_model <- load_model(model_output_path) # load the model

if (Sys.getenv("WEEKLY_EVALUATION", "false") == "true") {
  prices <- predict_premium(trained_model, x_clean)
  write.table(x = prices, file = output_prices_file, row.names = FALSE, col.names = FALSE, sep = ",")
} else {
  claims <- predict_expected_claim(trained_model, x_clean)
  write.table(x = claims, file = output_claims_file, row.names = FALSE, col.names = FALSE, sep = ",")
}
# 
# test <- x_clean 
# test$claim <- claims
# test$prices <- prices
# test <- test %>% mutate(ratio = claim / prices)
# test %>% group_by(ajust_n_claim)   %>% summarise( across(c(claim, prices), sum)) %>% mutate(ratio = claim/prices)
# test %>% 
#   ggplot(aes(x= ratio))+ 
#   geom_density()+
#   facet_wrap(~ajust_n_claim, scales = "free")
# 
# 
# test %>%   ggplot(aes(x= ratio))+ 
#   geom_density()
