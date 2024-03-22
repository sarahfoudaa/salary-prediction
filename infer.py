import numpy as np

def pipeline(test_array):
  '''
  Function that receives the raw data from the user to process it to a format that the model can digest 
    it perform label encoding to the designation, log transform the age, keeps the past_exp and years_months as they are, one-hot encode the unit

  Parameters: 
    test_array (list): list of size 5 that contains the designation, age, past_exp, years_months, and unit respectively 
 
  Return:
    features (list): a list of size 10 of the received list,test_array, after transformation 
  '''
  DESIGNATION = ['Analyst','Senior Analyst','Associate','Manager','Senior Manager','Director'] # get index
  index = DESIGNATION.index(test_array[0])
  des = index

  age = np.log(test_array[1])

  past_exp = test_array[2]

  years_months = test_array[3]
  UNIT_CAT = ['Finance','IT','Management',	'Marketing','Operations','Web']
  UNITZeroes = [0,0,0,0,0,0]
  index = UNIT_CAT.index(test_array[4])
  UNITZeroes[index] = 1
  unit = UNITZeroes

  features  = [des, age, past_exp, years_months]
  features = features + unit
  return features
