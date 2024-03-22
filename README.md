# Salary Prediction regression project 


## Table of contents

* [Requirments](#Requirments)
* [Problem-statement](#Problem-statement)
* [Dataset](#Dataset)
* [Exploratory Data Analysis (EDA)](#EDA)
* [Feature-engineering](#Feature-engineering)
* [Feature-selection](#Feature-selection)
* [Pre-processing](#Pre-processing)
* [Model](#Model)
* [Resultls](#Results)

# Requirments
* lazypredict == 0.2.12
* pandas == 1.5.3
* seaborn == 0.13.1
* pathlib == 1.0.1
* matplotlib == 3.7.1
* numpy == 1.25.2
* sklearn == 1.2.2
* scipy == 1.11.4
* pickle == 4.0

# Problem-statement
Salaries in the field of data professions vary widely based on factors such as experience, job role, and
performance. Accurately predicting salaries for data professionals is essential for both job seekers and
employers.

# Dataset
* `FIRST NAME`: First name
* `LAST NAME`: Last name
* `SEX`: Gender
* `DOJ`: Date of joining the company
* `CURRENT DATE`: Current date of data
* `DESIGNATION`: Job role/designation
* `AGE`: Age
* `SALARY`: Target variable, the salary of the data professional
* `UNIT`: Business unit or department
* `LEAVES USED`: Number of leaves used
* `LEAVES REMAINING`: Number of leaves remaining
* `RATINGS`: Ratings or performance ratings
* `PAST EXP`: Past work experience

* total number of records -->(100%) 2640

* train -->(80%) 

  * train-->(70%)
  
  * eval-->(10%)

* test-->(20%)

# EDA
The Exploratory data analysis step is the first step after collecting and importing the dataset and it is crucial for any machine learning project, it helps identify genral patterns in the data, explore the data to decide in the method to be carried on in the next step, the feature engineering step

a custum function is writen to display some informations about the dataset like a sample of the dataset. info, the describtion, number of unique values in each feature, sum and percentage of the missing values in each feature, the skewness of each feature,and a heatmap that visualize the missing values compared with the present values in each feature

datetime features --> 'DOJ', 'CURRENT DATE' 

numerical features 
* continous --> 'AGE', 'SALARY', 'LEAVES USED', 'LEAVES REMAINING', 'PAST EXP'
* discrete --> 'RATINGS'

categorical feature --> 'SEX', 'DESIGNATION', 'UNIT'

features with missing values --> 'LAST NAME', 'DOJ', 'AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS'

the features 'LEAVES USED' and 'LEAVES REMAINING' when added row wise have the sum of 30 so

a custom written function -vis- receives a feature and then displayes its histplot, boxplot, and violinplot to have an insigt about the distribution and outliers in this feature 

features with outliers -->  'AGE' ,'SALARY', 'PAST EXP'

a custom function is writen to detect the distribution according to the following threshold
* features that are highly positively skewed (skewness >1) 
* features that are highly negatively skewed (skewness <-1)
* features that are moderately positively skewed (0.5 < skewness < 1 ) 
* features that are moderately negatively skewed (-0.5 < skewness < -1 ) 
* features that are normally distributed (-0.5 < skewness < 0.5 )

positively skewed --> 'AGE', 'SALARY', 'PAST EXP'

normaly distributed --> 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS'

features that skewed --> 'AGE', 

'PAST EXP' and 'SALARY' are positivily correlated 

'AGE' and 'SALARY' are positivily correlated
 

# Feature-engineering
The feature engineering step is the second step after the EDA where based on the insights gained from the EDA regarding the features. in this step outliers and missing values are handeled, encoding of the necessary features take place, and normalization and standerdization if required 

* handelling missing values
  'AGE' --> mean is used beacause AGE is a continuous feature and the missingness is assumed to be missing at random

 'RATINGS' --> mode is used because RATINGS is a categorical variables or discrete variables with a few distinct values

'FIRST NAME' and 'LAST NAME' are dropped as they are usless in this case

* Type casting
  a custom function is writen to type cast features that have the wrong datatype
  'AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS' are converted to int
  'DOJ','CURRENT DATE' are converted into datetime

two new features are added 'nb_months', 'yearsMonths' using a custom written function that takes the date the job stated and ended and calculate how much months 'nb_months', and how much years 'yearsMonths' this employee has been in this position 
 
* handelling outliers
  a custom written function to handel features with outliers ('AGE', 'PAST EXP') is called to transform the aoutliers to be the the range of the IQR of the feature
* AGE  
![image](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/46d65e79-ed4d-4f9e-9b50-b33d3aa2958d)
![download](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/6bfe5cca-3082-41b6-8bb3-f623376b1e07)

* PAST EXP
![image](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/154da058-33c1-4188-b372-d5c3e4634623)
![download](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/cdea5abc-d578-4233-aaa1-59f668b8cad8)


* Removing duplicated rows with the help of a custom written function

* Feature encoding
  after experimenting with label encoding using sklearn and hard coded encoding the hard coded performs better so a function was written to perfrom label encoding
 'DESIGNATION' --> ordinal encoding --> label encoded beacuse it is a categorical feature and based on rank
 'SEX', 'UNIT' --> nominal encoding --> one-hot encoded beacause they are categorical feature, not ranked, doesnt have so many categories(doesn't exced 10 category)

* log transformation 
'AGE' and 'SALARY' are positivly skewed so log transformation is perfomrd on them
after ploting 'nb_months' and 'yearsMonths' they needed to be log tranformed as well
before vs after 
![download](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/d5e0ce91-4138-4600-9071-e55db66c5a37)

![download](https://github.com/sarahfoudaa/salary-prediction/assets/87505343/036348ec-f689-4cb4-86c0-49567a7c770f)





# Feature-selection

# Model

# Results
* MAE: 0.06749809805980801
* MSE: 0.006655037272039222
* RMSE: 0.08157841180140259
* R2 Square 0.9478234217946953
