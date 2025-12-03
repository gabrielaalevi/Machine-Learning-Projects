## Project Overview

In this project, we work with a datased that correlates social media usage, sleep quality, stress levels and exercise habits to a Happiness Index, which ranges from 1 to 10. We aim to build a Machine Learning that can predict a person's happiness levels by analysing their routines. 

The dataset contains the following columns:

- **User_ID** (int): number to designate each participant in the survey.
- **Age** (int): participant's age.
- **Gender** (str): participant's gender (Male, Female or Other).
- **Daily Screen Time (hrs)** (int): average number of hours the participant uses digital screens.
- **Sleep Quality** (int): how the participant rated their sleep quality, ranging from 1 (terrible) to 10 (perfect).
- **Stress Level** (int): how the participant rated their stress level, ranging from 1 to 10. In the source material, there is no explanation on how the score was calculated. Therefore, we do not know for sure if 1 is a very low stress level or a very high stress level. For this work, we will assume 1 is a terrible stress level, and 10 is very low stress.
- **Days without Social Media** (int): largest period the user has gone without checking their social media profiles, in days.
- **Exercise Frequency** (int): number of times the participant exercises in a week.
- **Social Media Platform** (str): participant's most used social media platform.
- **Happiness Index** (int): how the participant rated their happiness level, ranging from 1 (very unhappy) to 10 (very happy). This is our target feature to predict.

## Project Outline

### Preprocessing

In this phase, we standardize column names and string entries, transforming them to lowercase and replacing blank spaces by underscores. We drop the 'User ID' column, since it is not relevant for our analysis. Lastly, we set the 'Gender' and 'Social Media Platform' to the 'category' type.

### Exploratory Data Analysis

We begin our analysis by checking if there are any missing values in our dataset, or any duplicated rows. We find no missing values or duplicated rows. We then take a look at our numerical variables and see if they agree with our common sense, as to avoid errors. The maximum amount of times a participant exercised in a week was 7, which sounds possible. All the variables that range from 1 to 10 are between these values, as seen by their minimum and maximum values. The participants' ages vary from 16 to 49, which is reasonable. Therefore, our dataset seems to be ok, and not present any errors caused by wrong values.

Then, we check the unique values present in our categorical features. This information will be relevant in next sessions, when we need to encode this information. We see 3 options for gender (Male, Female and Other), and many different options for the preferred Social Media Platform. 

We follow by performing an univariate analysis, to understand how our variables are distributed in our dataset. Our main takeaways are:

- **Age** (Mean: 32.98, Median: 34, Standard Deviation: 9.96, Skewness: -0.12) follows an almost uniform distribution, with no outliers. Younger users may be more prone to high social media usage, and therefore may present a lower happiness index. This could be an important feature for our predictive models. This relation will be further analysed soon. The only necessary transformation for this data is scaling/standardizing.
- **Daily Screen Time** (Mean: 5.53, Median: 5.6, Standard Deviation: 1.73, Skewness: 0.03) follows an almost normal distribution, confirmed by the graph and the low skewness value. We see one outlier with screen time higher than 10 hours a day, which may be interesting to remove from the dataset. In a real-life situation, it would be interesting to separate screen time at work (doing work related tasks), and screen time at leisure time. Participants that work with computers may show a high daily screen time, but not necessarily a high social media usage. It would be interesting to see if it is only social media usage that impacts the participant's happiness levels, or if it is the daily screen time as a whole.
  
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
