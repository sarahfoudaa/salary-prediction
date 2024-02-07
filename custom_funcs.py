import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pylab
from config import data_path
from sklearn import preprocessing

#--------------------------EDA functions--------------------------
def data_info(df):
  '''
  Function that prints some important informations about the raw dataset at the beginning of the EDA select_dtypes
  
  parameters: 
    df(DataFrame):dataframe of the raw dataset 
  '''
  print('sample of the data \n', df.head(5),'\n')
  print('data types \n', df.info(),'\n \n')
  print('data describtion \n', df.describe().T,'\n \n')
  print('unique values for each feature \n', df.nunique(),'\n \n')
  print('sum of missing values \n',df.isna().sum(),'\n \n')
  print('percentage of missing values \n', df.isna().mean(),'\n \n')
  print('skewness of numerical data', df.skew())
  sns.heatmap(df.isna(), cbar = False, cmap = 'viridis')


def vis(df,feature):
  '''
  Function that visualize the histplot, boxplot, and violinplot for a specific feature in the dataset to give insight on the distribution, the IQR, and if there is any outliers
  
  parameters:
    df(DataFrame):dataframe of the dataset
    feature(string): a column in the sent received dataset
  '''
  sns.set()
  plt.figure(figsize = (7,5))
  fig, axes = plt.subplots(2, 2,figsize = (15,8))
  sns.histplot(x = df[feature],ax=axes[0,0],kde = True,stat = 'density')#https://www.geeksforgeeks.org/how-to-make-histograms-with-density-plots-with-seaborn-histplot/
  sns.boxplot(data = df, x= feature,ax=axes[0,1]) #https://www.labxchange.org/library/items/lb:LabXchange:46f64d7a:html:1
  sns.violinplot(x = df[feature],ax=axes[1,0])


def detect_outliers_iqr(df,features):
  '''
  Function that calculates the lower and upper bounf of a nuerical feature to then get the outliers that are not within these bounds and prints these outliers values
  
  Parameters:
    df(DataFrame):dataframe of the dataset
    feature(string): a column in the sent received dataset
  '''
  outliers = []
  before_lwr_bound= []
  after_upr_bound =[]
  for feature in features:
    df[feature] = sorted(df[feature])
    q1 = np.percentile(df[feature], 25)
    q3 = np.percentile(df[feature], 75)
    IQR = q3- q1
    lwr_bound = q1 - (1.5*IQR)
    upr_bound = q3 + (1.5*IQR)

    for i in df[feature]:
      if (i<lwr_bound):
        outliers.append(i)
        before_lwr_bound.append(i)
      if (i>upr_bound):
        outliers.append(i)
        after_upr_bound.append(i)
    print("-------------------",feature,"-------------------")
    print("Outliers of ",feature ," from IQR method: ", outliers)
    print('before lower bound of ',feature ," : ", before_lwr_bound)
    print('after upper bound of ',feature ," : ", after_upr_bound)
    sns.histplot(x =outliers)
    plt.show()


def detect_distriution(df):
  '''
  Function that detect the skewness from the normal distribution of the numerical features in the dataset and divid them into categories (highly_positively_skewed, highly_negatively_skewed, moderately_positively_skewed, moderately_negatively_skewed, and normally_distributed) which each of them have a range of skewness

  parameters:
    df(DataFrame):dataframe of the dataset

  Returns:
    non_normal_features(string): a string that contains all non uniform distributed features 
  '''
  print(df.skew())
  highly_positively_skewed = [x for x in df.select_dtypes(include = ['int', 'float']).columns if dict(df.skew(numeric_only=True)>1)[x] == True]
  highly_negatively_skewed = [x for x in df.select_dtypes(include = ['int', 'float']).columns if dict(df.skew(numeric_only=True)<-1)[x] == True]
  moderately_positively_skewed = [x for x in df.select_dtypes(include = ['int', 'float']).columns if (dict(df.skew(numeric_only=True)>0.5 )[x] == True) and (dict(df.skew(numeric_only=True)<1 )[x] == True)]
  moderately_negatively_skewed = [x for x in df.select_dtypes(include = ['int', 'float']).columns if (dict(df.skew(numeric_only=True)>-0.5 )[x] == True) and (dict(df.skew(numeric_only=True)<-1 )[x] == True)]
  normally_distributed = [x for x in df.select_dtypes(include = ['int', 'float']).columns if (dict(df.skew(numeric_only=True)>-0.5 )[x] == True) and (dict(df.skew(numeric_only=True)<0.5 )[x] == True)]
  print()
  print('features that are highly positively skewed (skewness >1) are ',highly_positively_skewed)
  print('features that are highly negatively skewed (skewness <-1) are ',highly_negatively_skewed)
  print('features that are moderately positively skewed (0.5 < skewness < 1 ) are ',moderately_positively_skewed)
  print('features that are moderately negatively skewed (-0.5 < skewness < -1 ) are ',moderately_negatively_skewed)
  print('features that are normally distributed (-0.5 < skewness < 0.5 ) are',normally_distributed)
  print()

  for col in df.select_dtypes(include = ['int', 'float']).columns:
    plt.subplot(1,2,1)
    sns.violinplot(df[col])
    plt.subplot(1,2,2)
    sns.histplot(df[col],kde = True,stat = 'density')
    plt.show()

  non_normal_features = highly_positively_skewed + highly_negatively_skewed + moderately_positively_skewed + moderately_negatively_skewed

  print('transformation ',non_normal_features,' to change the distribution')
  return non_normal_features


def plot_kde_probplot(df,feature,title):
  '''
  Function that plots the kdeplot and subplot to them give us insight of which transformationis better for this feature from the plotting and the calculations of the mean and std

  parameters:
    df(DataFrame):dataframe of the dataset
    feature(string): a column in the sent received dataset
    title(string): a string that indicates what transformation are we applying for visualization
  '''
  plt.figure(figsize = (5,3))
  plt.subplot(1,2,1)
  sns.kdeplot(df[feature])
  plt.subplot(1,2,2)
  stats.probplot(df[feature], plot = pylab)
  print('mean = ',df[feature].mean(),' std = ',df[feature].std())
  plt.title(title)
  plt.show()


def normality(df, feature):#original, log, reciprocal, square root, and exponential
  '''
  Function that calls for the plot_kde_probplot function to plot different types of transformations( log, reciprocal, square root, and exponential) for a scpecific feature to give us insight which is the best according to the ploting and calculations displayed 

  Parameters:
    df(DataFrame):dataframe of the dataset
    feature(string): a column in the sent received dataset
  '''
  #original
  plot_kde_probplot(df,feature,'original')
  print(df[feature].skew(),'\n')

  #log
  test = pd.DataFrame(np.log(df[feature].dropna()))
  plot_kde_probplot(test, feature,'log')
  print(test.skew(),'\n')

  #reciprocal
  test = pd.DataFrame(1/df[feature].dropna())
  plot_kde_probplot(test, feature,'reciprotical')
  print(test.skew(),'\n')

  #square root
  test = pd.DataFrame(np.sqrt(df[feature].dropna()))
  plot_kde_probplot(test, feature,'square root')
  print(test.skew(),'\n')

  #exponential
  test = pd.DataFrame(df[feature].dropna()**(1/1.2))
  plot_kde_probplot(test,feature,'exponential')
  print(test.skew(),'\n')



#--------------------------feature engineering funcitons--------------------------
def split(ratio):
  '''
  Function that splits a dataframe into train dataset and test according to a specific given ratio

  Parameters:
  ratio(int): the ratio the dataset is going to be splitted into

  Returns:
   df(DataFrame): a copy of the original dataset 
   X_train(DataFrame): train dataset that will be feature engineered
   X_test(DataFrame): test datset that wont be feature engineered as it will be used in the evaluation step to evaluate the model(S)
  '''
  df = pd.read_csv(data_path)
  X_train, X_test  = train_test_split(df, test_size = ratio, random_state = 41)
  X_train.reset_index(inplace = True, drop=True)
  X_test.reset_index(inplace = True, drop=True)

  return df,X_train, X_test


def splitModeling(X_train,y_train, ratio ):
  '''
  Function that splits the dataset after the feature engineering and feature selection steps into X_train, X_test, y_train, y_tes

  Parameters:
    X_train(Dataframe): clean train dataset
    y_train(Dataframe): dataset's label or target
    ratio(int): the ratio the dataset is going to be splitted into

  Returns:
    X_train(Dataframe): train datset that is ready for modeling 
    X_test(Dataframe): test datset that is ready for testing 
    y_train(Dataframe): label of train 
    y_test(Dataframe): label of test
  '''
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = ratio, random_state = 41)
  return X_train, X_test, y_train, y_test


def subtract_nan(row):
  '''
  Funciton that handels missing values in the two features LEAVES USED and LEAVES REMAINING 

  Parameters:
    row(Dataframe)

  Return:
    row(Dataframe)
  '''
  if np.isnan(row['LEAVES USED']):
    row['LEAVES USED'] = 30 - row['LEAVES REMAINING']
  if np.isnan(row['LEAVES REMAINING']):
    row['LEAVES REMAINING'] = 30 - row['LEAVES USED']

  return row


#after handelling all missing values
def type_casting(df,data_types_dict):
  '''
  Funciton that type cast all the features that not in the right datatype

  Parameters:
    df(dataframe): dataset that the needs type casting 
    data_types_dict(dictionary): a dictionary that contains datatype as the keys and corresponding to them the features to be casted
  '''
  if(data_types_dict['int']):
    df[data_types_dict['int']] = df[data_types_dict['int']].astype(int)

  for feature in df[data_types_dict['datetime']]:
    df[feature] = pd.to_datetime(df[feature])


def convert_dates_to_duration(df, starts, ends):
  '''
  Function the calculated the duration in between two given datset

  Parameters:
    df(Dataframe)
    starts(datetime): that start date to the event 
    ends(datetime): that end date to the event 
  '''
  df['nb_months'] = ((df[ends] - df[starts])/np.timedelta64(1,'M')).astype(int)
  df['yearsMonths'] = df['nb_months'].apply(lambda x: int(x/12) + (x%12)/12)
  df.drop(columns = ['DOJ', 'CURRENT DATE'],inplace = True)


def past_exp_binning(x):
  '''
  Funciton that segments or bins numerical feature into specific segments and change it into categorical 

  Parameters:
    x(int): the values to be replaced with a bins

  Returns:
    the bin in will be in 
  '''
  x = int(abs(x))
  if (x==0):
    return "0"
  elif (1 <= x <= 3):
    return "1~3"
  elif (4 <= x <= 9):
    return "4~9"
  elif (10 <= x <= 18):
    return "10~18"
  elif (19 <= x <= 23):
    return "19~23"


def outliers_handeler(df,features):
  '''
  Function that handels outliers and put them in the range of the IQR
  
  parameters:
    df(DataFrame)
    features(string)
  '''
  for feature in features:
    tenth_percentile = np.percentile(df[feature],10)
    ninetieth_percentile = np.percentile(df[feature],90)
    b = np.where(df[feature]< tenth_percentile, tenth_percentile, df[feature])
    b = np.where(b>ninetieth_percentile, ninetieth_percentile, b)
    df[feature] = pd.DataFrame(b)


def remove_duplicates(df):
  '''
  Function that removed duplicates 

  parameters:
    df(dataframe)
  '''
  print('nuber of duplicats = ',df[df.duplicated()==True].shape[0])
  print('assumed numbers of rows remaining = ',df.shape[0] - df[df.duplicated()==True].shape[0])
  df.drop_duplicates(keep = 'first', inplace = True)
  df.reset_index(inplace = True,drop=True)
  print('number of rows after removing the duplicates = ',df.shape[0])


def label_encoding(df,feature,categories):
  '''
  Funciton that label encode a feature accordning to specific order 

  Parameters:
    df(dataframe)
    feature(string)
    categories(list): a lsit of the categories in the received feature that wil be replaced in the same order
  '''
  for i,cat in enumerate (categories):
    df[feature].replace({cat:i},inplace = True)
