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

In this session, we standardize column names and string entries, setting them all to lowercase and removing blank spaces. We drop the 'Customer Id' and 'Surname' columns, since they are not relevant for machine learning algorithms. We check if there are any missing values in the dataset that may need to be handled, which is not the case. We also see there are no duplicated rows. Lastly, we convert all categorical variables to the 'category' type. We also convert discrete integer variables, such as 'Exited', 'Complained, and 'Satisfaction Score', to categorical as well.

### Exploratory Data Analysis

We analyse the dataset to understand its main characteristics and relations between features. Firstly, we begin by plotting frequency plots (for categorical variables), histograms and box plots (for numerical variables), to understand the variability in our data. This analysis allows for some conclusions:

- **Credit Score** (Mean: 650.53, Median: 652.0, Skewness: -0.07, Standard Deviation: 96.65) has a nearly normal distribution, which is confirmed by the low skewness. We see there are a few outliers with low credit score, which could be an indicator for churning. The only transformation required for this feature is scaling/standardizing.
- **Age** (Mean: 38.92, Median: 37.0, Skewness: 1.01, Standard Deviation: 10.48) has a highly skewed distribution, showing the client dataset tends towards younger customers between 30 and 40 years old. There could be a relation between age and churning, with younger clients having a larger tendency to exit the bank. It is necessary to apply a logarithmic transformation to the data before standardizing, to reduce skewness.
- **Tenure** (Mean: 5.01, Median: 5.0, Skewness: 0.01, Standard Deviation: 2.89) is uniformally distributed, showing even spread among the possible values. It is necessary only to standardize the data.
- **Balance** (Mean: 76485.89, Median: 97198.54, Skewness: -0.14, Standard Deviation: 62397.40) shows a huge peak at zero, showing many accounts have a null balance. This could be a huge predictor of churning, since it points to clients that do not use their account. The amount of zero-balanced accounts reduce the mean and create a discrepancy between the mean and the median. 
