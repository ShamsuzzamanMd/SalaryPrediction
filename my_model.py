import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle

data = pd.read_csv("hiring.csv")
#data.head()
print(data)

# Missing values removal
data['experience'].fillna(0, inplace = True)
data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean(), inplace = True)

print(data)



X = data.iloc[:, :3]
#print(x)



# Words to integer
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = data.iloc[:, -1]



#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))