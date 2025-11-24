## Project Overview

In this project, we work with the client dataset from a fictional bank. Our goal is to predict which clients have left the bank (i.e. churned) and which remained, using data such as
credit score, location, tenure duration, account balance and etc. This allows the bank to implement retention programs targeted to clients with higher risk of leaving.

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

In this session, we standardize column names and string entries, setting them all to lowercase and removing blank spaces. We drop the 'Customer Id' and 'Surname' columns, since
they are not relevant for machine learning algorithms. We check if there are any missing values in the dataset that may need to be handled, which is not the case. Lastly, we convert
all categorical variables to the 'category' type. We also convert discrete integer variables, such as 'Exited', 'Complained, and 'Satisfaction Score', to categorical as well.

### Exploratory Data Analysis

We sistematically analyse the dataset to understand its main characteristics and relations between features.
