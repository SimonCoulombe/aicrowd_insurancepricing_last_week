AICrowd insurance pricing competition last week submission
================
Simon Coulombe
Marc 6th 2021

# aicrowd\_insurancepricing\_starterpack

A starterpack for the insurance pricing competition at
<https://www.aicrowd.com/challenges/insurance-pricing-game>

It uses the {recipes} package to prepare the data, ensuring that you
will always have the same features at the end.

I used a “tweedie” model, might be new for some folks. It allows you to
model claim amount directly and is an alternative to using a frequency +
a severity model

I also used the “recipes” package, which insure that you won’t create
extra dummy variables by mistake.

I purposefully set the hyperparameters to something absolutely stupid. I
also didnt do any clever feature engineering.

TO USE: run fit\_model() on the training data to generate
trained\_model.xgb, then zip the whole folder and upload as a submission

fit\_model() will search for the best hyperparameters by calling the
fit\_and\_evaluate\_model() function for many model hyperparameters. the
fit\_and\_evaluate\_model uses 100% of the data as a test set to better
estimate model performance. once this search is complete, we will
finally fit\_most\_promising\_model() and call it a day!
