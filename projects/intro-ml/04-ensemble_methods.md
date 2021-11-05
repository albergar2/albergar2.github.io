---
layout: post
title: "Supervised Machine Learning: Ensemble Methods"
author: Alberto García Hernández
date: "01/01/2021"
---

[Back](../../)
#### [Download Notebook](https://github.com/albergar2/data_science_material/blob/master/ML/supervised/04-ensemble_methods.ipynb)


**Ensemble methods** combine several decision trees to produce better predictive performance than utilizing a single decision tree. The main principle behind the ensemble model is that a group of weak learners come together to form a strong learner.

1. **Bagging (Bootstrap Aggregation)** is used when our goal is to reduce the variance of a decision tree by creating several subsets of data from the training sample chosen randomly with replacement. Now, each collection of subset data is used to train their decision trees. As a result, we end up with an ensemble of different models. Average of all the predictions from different trees are used which is more robust than a single decision tree.

2. **Boosting** new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models. Boosting algorithms convert a set of weak learners into a single strong learner by initializing a strong learner (usually a decision tree) and iteratively creating a weak learner that is added to the strong learner. They differ on how they create the weak learners during the iterative process.


```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
```

## 1. Load Data


```python
wine = load_wine()
X = pd.DataFrame(wine.data, columns=iris.feature_names)
y = wine.target
```


```python
X.head()
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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2])




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

## 2. Random Forest

Random Forest is an ensemble machine learning algorithm that follows the bagging technique whose base estimators are decision trees. Random forest randomly selects a set of features that are used to decide the best split at each node of the decision tree.

1. Random subsets are created from the original dataset (bootstrapping).
2. At each node in the decision tree, only a random set of features are considered to decide the best split.
3. A decision tree model is fitted on each of the subsets.
4. The final prediction is calculated by averaging the predictions from all decision trees.

1. **n_estimators**: It defines the number of decision trees to be created in a random forest.
2. **criterion**: "Gini" or "Entropy."
3. **min_samples_split**: Used to define the minimum number of samples required in a leaf node before a split is attempted
4. **max_features**: It defines the maximum number of features allowed for the split in each decision tree.
5. **n_jobs**: The number of jobs to run in parallel for both fit and predict. Always keep (-1) to use all the cores for parallel processing.


```python
clf = RandomForestClassifier(n_estimators=100, 
                             criterion='gini', 
                             min_samples_split=5, 
                             max_features=4, 
                             n_jobs=-1)
clf.fit(X_train, y_train)
```




    RandomForestClassifier(max_features=4, min_samples_split=5, n_jobs=-1)




```python
clf.score(X_test, y_test)
```




    0.9777777777777777



## 3. AdaBoost

At each iteration, **Adaptive Boosting** changes the sample distribution by modifying the weights attached to each of the instances. It increases the weights of the wrongly predicted instances and decreases the ones of the correctly predicted instances. The weak learner thus focuses more on the difficult instances. After being trained, the weak learner is added to the strong one according to his performance (so-called alpha weight). The higher it performs, the more it contributes to the strong learner.


```python
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
```




    AdaBoostClassifier(n_estimators=100, random_state=0)




```python
clf.score(X_test, y_test)
```




    0.9555555555555556



## 4. Gradient Boosting

**Gradient boosting** trains the weak learner on the remaining errors (so-called pseudo-residuals) of the strong learner. It is another way to give more importance to the difficult instances. At each iteration, the pseudo-residuals are computed and a weak learner is fitted to these pseudo-residuals. Then, the contribution of the weak learner (so-called multiplier) to the strong one is computed by using a gradient descent optimization process. The computed contribution is the one minimizing the overall error of the strong learner.


```python
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
```




    GradientBoostingClassifier(random_state=0)




```python
clf.score(X_test, y_test)
```




    0.9555555555555556



## 5. XGBoost

Extreme Gradient Boosting (XGBoost) is a more efficient version of gradient boosting framework containing both a linear model solver and tree learning algorithms. The reason behind it’s efficiency is it’s capacity to do parallel computing on a single machine. 

The problem with general boosting was
- Can’t extract the linear combination of features
- Small predictive power (high variance)

Gradient boosting approach:
- Control tree structure (maximum depth, minimum samples per leaf),
- Control learning rate (shrinkage),
- Reduce variance by introducing randomness (stochastic gradient boosting – using random subsamples of instances and features)

XGBoost improved it with some good features like:
- Good bias-variance (simple-predictive) trade-off “out of the box”
- Great computation speed

XGBoost’s objective function is a sum of a specific loss function evaluated overall predictions and a sum of regularization term for all predictors (KK trees).


```python
model = XGBClassifier()
model.fit(X_train, y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=0, num_parallel_tree=1,
                  objective='multi:softprob', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




```python
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
```




    0.9555555555555556




```python

```
