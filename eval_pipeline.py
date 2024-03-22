from custom_funcs import type_casting
from custom_funcs import convert_dates_to_duration
import numpy as np
from infer import pipeline
from sklearn import metrics

def testing(df):
  '''
  Function that receives the raw data as it is in the original test dataset and take only some of the features as selected from the feature selection step, transform the wrong datatypes, and add the new features add in the feature engineering step

  Parameters: 
    df(DataFrame): raw test dataset as received 
 
  Return:
    x(array of lists): tranformed dataset that is ready for evaluation
    y(array): 
  '''
  x = df[['DOJ',	'CURRENT DATE',	'DESIGNATION',	'AGE',	'UNIT', 'PAST EXP']]
  y = df['SALARY']

  data_types_dict = {
    'int': ['AGE'],
    'datetime': ['DOJ','CURRENT DATE']
  }
  type_casting(x,data_types_dict)
  convert_dates_to_duration(x,'DOJ','CURRENT DATE')

  x = x[['DESIGNATION', 'AGE', 'PAST EXP', 'yearsMonths', 'UNIT']]

  x = x.apply(pipeline, axis = 1)
  y = np.log(y)
  return x,y


def print_evaluate(true, predicted):
    '''
    Functiom that prints the MAE, MSE, RMSE, and R2 score 

    Parameters:
      true(series): a series that holds the true values or of the test dataset 
      predicted(series): a series that holds the predected values or of the test dataset after prediction 

    '''
    mae, mse, rmse, r2_square = evaluate(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

def evaluate(true, predicted):
    '''
    Function that calculates the mean avg error, mean square error, root mean square error and r2 sqaure 

    Parameters:
      true(series): a series that holds the true values or of the test dataset 
      predicted(series): a series that holds the predected values or of the test dataset after prediction 
  
    Return:
      mae(int): mean avg error
      mse(int): mean square error
      rmse(int): root mean square erro
      r2_square(int): r2 sqaure 
    '''
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square
