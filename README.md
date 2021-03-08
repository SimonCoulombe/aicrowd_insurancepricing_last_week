AICrowd insurance pricing competition last week submission
================
Simon Coulombe
March 6th 2021

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

This is the code I used to generate my submission for the last week of
the AICrowd insurance pricing competition.

My model is nothing fancy, but here is a quick look around it.

It’s basically an xgboost using the “tweedie” objective function like I
described ealier in my [starter
pack](https://github.com/SimonCoulombe/aicrowd_insurancepricing_starterpack).
I considered doing a “poisson x gamma” model, or even a “logistic x
gamma” model, but I didnt have the time/will for that.

I used “rmse” as the eval\_metric to decide when to stop churning out
trees. I didnt check if using the loglikelihood was better.

My recipe has a bit more meat than in the starterpack. See function
train\_my\_recipe() in the [fit\_model.R
file](https://github.com/SimonCoulombe/aicrowd_insurancepricing_last_week/blob/main/fit_model.R).  
A couple “clever” things I did in that recipe:

-   concatenate population and town\_surface\_area to create a unique
    “town\_id”  
-   create “vh\_current”value", which is the value of the vehicle
    depreciated at a rate of 20% per year.

``` r
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
```

For the hyperparameters, I did some random grids and even tried some
bayesian search, but in the end I went with a set of parameters that
“felt” right:

    ## Rows: 1
    ## Columns: 11
    ## $ booster                <chr> "gbtree"
    ## $ objective              <chr> "reg:tweedie"
    ## $ eval_metric            <chr> "rmse"
    ## $ tweedie_variance_power <dbl> 1.66
    ## $ gamma                  <dbl> 4
    ## $ max_depth              <dbl> 4
    ## $ eta                    <dbl> 0.01
    ## $ min_child_weight       <dbl> 10
    ## $ subsample              <dbl> 0.6
    ## $ colsample_bytree       <dbl> 0.4
    ## $ tree_method            <chr> "hist"

For the pricing, I tried a very basic approach of a 20% profit margin
and a minimum price of 25$ for about half of the 10 practice weeks. For
the other half, I used the same model, but I set a random profit margin
between 1 and 100%. This is another “clever” thing I tried ;) To set a
random profit margin, I looked at the first 2 decimals of the predicted
claims (say 13 f the price was 92.13$) and decided that was the profit
margin. When setting the random prices, the idea was that I would
inflate the prices using the determined profit margin, then set the
decimals of the price to whatever the profit margin was so that I could
calculate my profit and conversion rate by profit margin. The idea was
to be able to tell what would be my market share at any profit margin
using only a single week of data, but we were never provided enough
weekly feedback to do this. Another “cunning plan” was to hide
information from the weekly test in the decimals of the prices I charge.
For example, a price of 109.23125 could have mean “23” percent profit
margin, “1” means “man” and “25” means 25 year old. That cunning plan
was also defeated since we were never given detailed enough information
:)

Looking at the conversion\_rate and the average profit of the
often/sometimes/never sold quotes for the weeks using random profit
margins, I have a feeling that around 20% was a good number. For
example, if you look at my “financials by conversion rate” table in the
week 8 feedback, then you see that the higher the profit margin, the
less often I sell a quote (obviously). You also see that my highest
profit per policy was for the policies sold “sometimes”, and the average
profit margin for that group was 21%.

Financials By Conversion Rate, week 8

    ## # A tibble: 4 x 6
    ##   `Policies won`      claim_frequency premiums conversion_rate `profit per poli…
    ##   <chr>                         <dbl>    <dbl>           <dbl>             <dbl>
    ## 1 often: 34.0 - 100.…            0.09     92.9            0.74            -24.2 
    ## 2 sometimes: 1.4 - 3…            0.1     132.             0.11              1.9 
    ## 3 rarely: 0.1 - 1.4%             0.1     143.             0.01              0.19
    ## 4 never:                         0.11    221.             0                 0   
    ## # … with 1 more variable: profit_margin <dbl>

In the final weeks I tried a random profit margin between 20-45%, the
idea being to try to get a few “very profitable policies” by trying to
sell at 30+% profit margin, while also ensuring that I sell at least a
few policies and remain on the leaderboard thanks to the 20-30% profit
margin on half the quotes. That didnt work very well.

My model has been profitable for the first half of the competition with
a very small market share, until people caught on that you needed to
have high profit margins.

I’ve been meaning to do target encoding for “town\_id” and
“vh\_make\_model” but never found the will/time to do it either. 

For the hyperparameters, I also tried to do something “clever” and use
100% of the training data as a test set.Here are the steps as seen in
fit\_and\_evaluate\_model() : \* create 5 20% folds. \* do xgb.cv on
combined data from fold A B C D to find optimal number of iterations \*
train model on combined A B C D with optimal number of iterations \*
predict on fold E. (the holdout folds) \* then repeat 4 more times,
using the different folds as holdout. I don’t think it was worth it, and
it meant runing everything 25 times for each set of hyperparameters. My
next idea was to evaluate the performance on fold E for all
hyperparameters, then estimate on fold A,B,C and D for the top 5
hyperparameters sets to pick the best one. I didnt have the will to do
it though.

HEre are the links to my weekly feedbacks

[week
10](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-10/0f9f9bb8-cdd1-4bbc-bbcc-6a92fda83aa5.html)
(20-45% profit margin)  
[week
9](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-9/aac0836d-c99d-462d-8da2-69c64237e604.html)
(20-45% profit margin)  
[week
8](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-8/78813c62-115a-4c34-86cd-a7688b62c0a2.html)
(1-100% profit margin)  
[week
7](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-7/34fc144a-ef88-40e3-885e-f6ec8d23359e.html)
(20% profit margin)  
[week
6](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-6/20d6f746-98b3-4fd9-8d2e-8050784e5adc.html)
(1-100% profit margin)  
[week
5](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-5/17190f91-87e3-4e42-bc9f-c5c73ed109f9.html)
(20% profit margn)  
[week
4](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-4/f1f63a62-5573-4662-a57b-fc90751ec547.html)
(20% profit)  
[week
3](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-3/4349b127-93f7-441f-aaca-956ab6283bed.html)  
[week
2](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-2/3ce37802-637c-4558-942c-311bd69dd7a1.html)  
[week
1](https://imperial-college-insurance-pricing-game.s3.eu-central-1.amazonaws.com/week-1/e2af6629-2ac6-44f8-87a6-6751d32d24a6.html)
(20% profit)

fit\_model() will search for the best hyperparameters by calling the
fit\_and\_evaluate\_model() function for many model hyperparameters. the
fit\_and\_evaluate\_model uses 100% of the data as a test set to better
estimate model performance. once this search is complete, we will
finally fit\_most\_promising\_model() and call it a day!
