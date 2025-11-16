## Project Overview

In this project, we aim to study how social media usage is related to life satisfaction. The dataset employed correlates social media usage, sleep quality and exercise habits to a Happiness Index, which ranges from 1 to 10. After analysing the data, we build a Machine Learning model to predict this happiness index using the features cited previously.

### Preprocessing

In this stage, we begin by reducing all column names and text features to lowercase, and removing spaces between words. We then rename some columns to shorter titles, to make it simpler to type them in our code. Lastly, we check for missing values, which there are none.

### Exploratory Data Analysis

We check all the unique values in our features. This is important for categorical features, which we will need to encode in the next sessions. We plot the number of instances and their values for each feature, and check almost all of them follow a normal distribution. In special, the happiness index distribution has more high life satisfaction values than lower satisfaction instances, which will be relevant in next sessions.

Next, we study how happiness index varies with each feature. We see the daily screen time (in hours), the sleep quality index, the stress levels and the frequency of exercising are good features to predict the happiness index. Meanwhile, age and the maximum number of days without social media are barely correlated with happiness index. This is supported by the correlation plot we do in the next cell.

With this information, we choose to drop the 'age' feature from the dataset, since it is the one with the smallest correlation with our target feature.

### Label Encoding

We encode the 'gender' and 'social media platform' features using one-hot encoding, so our machine learning model can employ them.

### Data Split

We split the data into training and testing sets. In the EDA session, we saw the happiness index column presents more high satisfaction values than lower satisfation instances. This could be a problem for our model, since it is possible the training set could have very few instances with low life satisfaction, and therefore our model would be biased to predict higher values for happiness index. So, we stratify our sets by happiness index, to guarantee each set has around the same proportion of lower happiness levels.

All the numerical features we employ in the model range between 1 and 10, therefore there is no need to scale our data.

### Model Training

Finally, we train our models. We employ RMSE and NRMSE to evaluate our models, since this is the best metric for regression problems without outliers. We also employ cross-validation for all models, and grid search to fine tune some hyperparameters. The models trained are:

- Linear Regression: RMSE of 0.9666.
- Linear Regression with Ridge Regularization: RMSE of 0.9664.
- Linear Regression with Lasso Regularization: RMSE of 0.9559.
- Decision Tree Regressor: RMSE of 1.0449.
- Random Forest Regressor: RMSE of 0.9645.
- Gradient Boosting Regressor: RMSE of 0.955.

Therefore, we see Gradient Boosting Regressor was the best model in the training set. 

### Hyperparameter fine-tuning

The grid search done in the previous session showed the best hyperparameters for the Gradient Boosting Regressor. Now, we do another grid search for values close to the ones selected in the previous session, to fine-tune the model. With this, we reduce the RMSE in the training test to 0.9546. Finally, we use this model to make predictions in the test set, and achieve a RMSE of 0.8999. The model has a better performance in the test set, therefore we do not need to worry about overfitting.
