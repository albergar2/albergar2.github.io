---
layout: post
title: "Supervised Machine Learning: Naive Bayes"
author: Alberto García Hernández
date: "01/01/2021"
---

[Back](../../)
#### [Download Notebook](https://github.com/albergar2/data_science_material/blob/master/ML/supervised/06-naive_bayes.ipynb)


Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Now, with regards to our dataset, we can apply Bayes’ theorem in following way: 

P(y|X) = {P(X|y) P(y)}/{P(X)}

where, y is class variable and X is a dependent feature vector (of size n) where: 

X = (x_1,x_2,x_3,.....,x_n)

## 1. Naive Bayes Classification

1. We assume the features are independent
2. Each feature is given the same importance (weight)


```python
# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
         'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

import pandas as pd
df = pd.DataFrame(data={'weather':weather,'temp':temp, 'play':play})
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weather</th>
      <th>temp</th>
      <th>play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rainy</td>
      <td>Mild</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rainy</td>
      <td>Cool</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
le = preprocessing.LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
df['temp'] = le.fit_transform(df['temp'])
df['play'] = le.fit_transform(df['play'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weather</th>
      <th>temp</th>
      <th>play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = df[['weather', 'temp']]
label = df['play']

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)
```




    GaussianNB()




```python
model.predict([[0,2]]) # 0:Overcast, 2:Mild
```




    array([1])




```python
model.score(features, label)
```




    0.7142857142857143



## 2. Gaussian Naive Bayes
Continuous values associated with each feature are assumed to be distributed according to a Gaussian (normal) distribution. This is as simple as calculating the mean and standard deviation values of each input variable (x) for each class value.

- Mean (x) = 1/n * sum(x)
- Standard deviation(x) = sqrt (1/n * sum(xi-mean(x)^2 ))


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
```


```python
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
```


```python
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```




    GaussianNB()




```python
gnb.score(X_test, y_test)
```




    0.9466666666666667




```python
y_pred = gnb.predict(X_test)
```


```python
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
```

    Number of mislabeled points out of a total 75 points : 4



```python

```
