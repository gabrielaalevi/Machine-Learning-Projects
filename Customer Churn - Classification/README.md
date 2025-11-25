## Project Overview

In this project, we work with the client dataset from a fictional bank. Our goal is to predict which clients have left the bank (i.e. churned) and which remained, using data such as credit score, location, tenure duration, account balance and etc. This allows the bank to implement retention programs targeted to clients with higher risk of leaving.

The dataset contains the following columns:

- **Customer Id** (int): Number to designate each client.
- **Surname** (str): Client's surname.
- **Credit Score** (int): Client's credit score in the bank.
- **Geography** (str): Country where the client resides.
- **Gender** (str): Client's gender.
- **Age** (int): Client's age.
- **Tenure** (int): Number of years the client's account has existed for.
- **Balance** (float): Client's account balance.
- **Number of Products** (int): Number of products the client has purchased through the bank.
- **Has Credit Card** (int): If client has a credit card (equals 1) or does not have a credit card (equals 0).
- **Is Active Member** (int): If the client is an active member of the bank (equals 1) or not (equals 0).
- **Estimated Salary** (float): Client's estimated salary.
- **Exited** (int): If the client has left the bank (equals 1) or not (equals 0). This is our target variable.
- **Complain** (int): If the client has complained to the bank customer services (equals 1) or not (equals 0).
- **Satisfaction Score** (int): Client's evaluation of their satisfaction with the bank services, varying from 1 (very displeased) to 5 (very pleased).
- **Card Type** (str): Client's card type. Possible values are Silver, Gold, Platinum or Diamond.
- **Points Earned** (int): Points earned by the client due to their credit card usage.

## Project Outline

### Pre-processing

In this section, we standardize column names and string entries, setting them all to lowercase and removing blank spaces. We drop the 'Customer Id' and 'Surname' columns, since they are not relevant for machine learning algorithms. We check if there are any missing values in the dataset that may need to be handled, which is not the case. We also see there are no duplicated rows. Lastly, we convert all categorical variables to the 'category' type. We also convert discrete integer variables, such as 'Exited', 'Complained, and 'Satisfaction Score', to categorical as well.

### Exploratory Data Analysis

We analyse the dataset to understand its main characteristics and relations between features. Firstly, we begin by plotting frequency plots (for categorical variables), histograms and box plots (for numerical variables), to understand the variability in our data. This analysis allows for some conclusions:

- **Credit Score** (Mean: 650.53, Median: 652.0, Skewness: -0.07, Standard Deviation: 96.65) has a nearly normal distribution, which is confirmed by the low skewness. We see there are a few outliers with low credit score, which could be an indicator for churning. The only transformation required for this feature is scaling/standardizing.
- **Age** (Mean: 38.92, Median: 37.0, Skewness: 1.01, Standard Deviation: 10.48) has a highly skewed distribution, showing the client dataset tends towards younger customers between 30 and 40 years old. There could be a relation between age and churning, with younger clients having a larger tendency to exit the bank. It is necessary to apply a logarithmic transformation to the data before standardizing, to reduce skewness.
- **Tenure** (Mean: 5.01, Median: 5.0, Skewness: 0.01, Standard Deviation: 2.89) is uniformally distributed, showing even spread among the possible values. It is necessary only to standardize the data.
- **Balance** (Mean: 76485.89, Median: 97198.54, Skewness: -0.14, Standard Deviation: 62397.40) shows a huge peak at zero, showing many accounts have a null balance. This could be a huge predictor of churning, since it points to clients that do not use their account. The amount of zero-balanced accounts reduce the mean and create a discrepancy between the mean and the median. In this case, we can apply the same principle behind a hurdle model: we create a new binary variable 'HasZeroBalance'. It is 1 for clients with zero balance, and 1 for clients with non-zero balances. This allows us to remove the zero balanced accounts from the Balance distribution. Subsetting the dataset to include only clients with non-zero balance, we find the Balance feature now presents a normal distribution, with 0.02 skewness. Therefore, after creating the 'HasZeroBalance' feature, we will only need to standardize the Balance column.
- **Estimated Salary** (Mean: 100090.24, Median: 100193.92, Skewness: 0.00, Standard Deviation: 57510.49) shows a highly uniform distribution. The client dataset does not favors clients of specific income. The only necessary transformation is standardization.
- **Points Earned**  (Mean: 606.52, Median: 605.0, Skewness: 0.01, Standard Deviation: 225.92) is also uniformly distributed, and does not present outliers. The only transformation necessary is standardization.
- **Geography** shows the majority of bank clients reside in France. It may be an interesting indicator for churning.
- **Gender** is almost evenly split, with a little bit more male clients than female clients.
- **Number of Products** has a high peak at 1, and decreases until 4. The majority of clients does not show intense engagement.
- **Has Credit Card** indicates more than 2/3 of the clients in the dataset have a credit card. The public without a credit card could present a high potential for churning.
- **Is Active Member** is evenly split in the dataset. Inactive clients are more likely to churn.
- **Exited** is our target variable. Around 20% of clients churn, while 80% remain in the bank. This representation imbalance could be harmful to our model, leading to an overrepresentation bias. It will be necessary to oversample the dataset, or at least stratify the training and testing sets.
- **Complain** shows 20% of clients complained to the bank. Clients complaining show a high potential for churning.
- **Satisfaction Score** is evenly distributed, with a small peak at 3. Low scores can be great predictors of churning, but the uniform distribution may harm its predictive power.
- **Card Type** is uniformly split, and is unlikely to be a high predictor of churning.
