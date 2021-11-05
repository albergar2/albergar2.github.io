---
layout: post
title: "Unsupervised Machine Learning: Market Basket Analysis"
author: Alberto García Hernández
date: "01/01/2021"
---

[Back](../../)
#### [Download Notebook](https://github.com/albergar2/data_science_material/blob/master/ML/unsupervised/01-market_basket_analysis.ipynb)


Tutorial: https://pythondata.com/market-basket-analysis-with-python-and-pandas/

Market basket analysis is the study of items that are purchased or grouped in a single transaction or multiple, sequential transactions. Understanding the relationships and the strength of those relationships is valuable information that can be used to make recommendations, cross-sell, up-sell, offer coupons, etc. It works by looking for combinations of items that occur together frequently in transactions.


```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

## 1. Load Data


```python
df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
```


```python
df.sample(5)
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>367030</th>
      <td>568831</td>
      <td>22080</td>
      <td>RIBBON REEL POLKADOTS</td>
      <td>5</td>
      <td>2011-09-29 11:35:00</td>
      <td>1.65</td>
      <td>16059.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>319288</th>
      <td>564846</td>
      <td>22297</td>
      <td>HEART IVORY TRELLIS SMALL</td>
      <td>24</td>
      <td>2011-08-30 15:16:00</td>
      <td>1.25</td>
      <td>14507.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>284430</th>
      <td>561871</td>
      <td>22720</td>
      <td>SET OF 3 CAKE TINS PANTRY DESIGN</td>
      <td>3</td>
      <td>2011-07-31 11:46:00</td>
      <td>4.95</td>
      <td>13018.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>173678</th>
      <td>551718</td>
      <td>90003C</td>
      <td>MIDNIGHT BLUE PAIR HEART HAIR SLIDE</td>
      <td>1</td>
      <td>2011-05-03 16:06:00</td>
      <td>3.73</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>322859</th>
      <td>565234</td>
      <td>85093</td>
      <td>CANDY SPOT EGG WARMER HARE</td>
      <td>2</td>
      <td>2011-09-02 09:38:00</td>
      <td>0.83</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Clean Data


```python
# In this data, there are some invoices that are ‘credits’ instead of ‘debits’ 
# so we want to remove those. They are indentified with “C” in the InvoiceNo field. 

df = df[~df['InvoiceNo'].str.contains('C')]
```

## 3. Analysis


```python
#We encode our data to show when a product is sold with another product. 
# If there is a zero, that means those products haven’t sold together. 

market_basket = df[df['Country'] =="United Kingdom"].groupby(
                ['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
```


```python
market_basket.head()
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
      <th>Description</th>
      <th>20713</th>
      <th>4 PURPLE FLOCK DINNER CANDLES</th>
      <th>50'S CHRISTMAS GIFT BAG LARGE</th>
      <th>DOLLY GIRL BEAKER</th>
      <th>I LOVE LONDON MINI BACKPACK</th>
      <th>NINE DRAWER OFFICE TIDY</th>
      <th>OVAL WALL MIRROR DIAMANTE</th>
      <th>RED SPOT GIFT BAG LARGE</th>
      <th>SET 2 TEA TOWELS I LOVE LONDON</th>
      <th>SPACEBOY BABY GIFT SET</th>
      <th>...</th>
      <th>wrongly coded 20713</th>
      <th>wrongly coded 23343</th>
      <th>wrongly coded-23343</th>
      <th>wrongly marked</th>
      <th>wrongly marked 23343</th>
      <th>wrongly marked carton 22804</th>
      <th>wrongly marked. 23343 in box</th>
      <th>wrongly sold (22719) barcode</th>
      <th>wrongly sold as sets</th>
      <th>wrongly sold sets</th>
    </tr>
    <tr>
      <th>InvoiceNo</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>536365</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>536366</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>536367</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>536368</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>536369</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4188 columns</p>
</div>




```python
# We want to convert all of our numbers to either a `1` or a `0` (negative numbers are 
# converted to zero, positive numbers are converted to 1).

def encode_data(datapoint):
    if datapoint <= 0: return 0
    if datapoint >= 1: return 1
    
market_basket = market_basket.applymap(encode_data)
```

The `apriori` function requires us to provide a minimum level of **support**. Support is defined as the percentage of time that an itemset appears in the dataset. If you set support = 50%, you’ll only get itemsets that appear 50% of the time. Setting the support level to high could lead to very few (or no) results and setting it too low could require an enormous amount of memory to process the data.


```python
itemsets = apriori(market_basket, min_support=0.03, use_colnames=True)
```


```python
itemsets.head()
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
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.045800</td>
      <td>(6 RIBBONS RUSTIC CHARM)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.031123</td>
      <td>(60 CAKE CASES VINTAGE CHRISTMAS)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.040336</td>
      <td>(60 TEATIME FAIRY CAKE CASES)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.046925</td>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.035140</td>
      <td>(ALARM CLOCK BAKELIKE PINK)</td>
    </tr>
  </tbody>
</table>
</div>



The final step is to build your **association rules** using the mxltend `association_rules` function. You can set the metric that you are most interested in (either `lift` or `confidence` and set the minimum threshold for the condfidence level (called `min_threshold`). The `min_threshold` can be thought of as the level of confidence percentage that you want to return. For example, if you set `min_threshold` to 1, you will only see rules with 100% confidence.

**Association analysis** uses a set of transactions to discover rules that indicate the likely occurrence of an item based on the occurrences of other items in the transaction.


```python
rules = association_rules(itemsets, metric="lift", min_threshold=0.5)
```


```python
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>0.049818</td>
      <td>0.046925</td>
      <td>0.030159</td>
      <td>0.605376</td>
      <td>12.900874</td>
      <td>0.027821</td>
      <td>2.415149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>0.046925</td>
      <td>0.049818</td>
      <td>0.030159</td>
      <td>0.642694</td>
      <td>12.900874</td>
      <td>0.027821</td>
      <td>2.659296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(GREEN REGENCY TEACUP AND SAUCER)</td>
      <td>(PINK REGENCY TEACUP AND SAUCER)</td>
      <td>0.050032</td>
      <td>0.037658</td>
      <td>0.030909</td>
      <td>0.617773</td>
      <td>16.404818</td>
      <td>0.029024</td>
      <td>2.517724</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(PINK REGENCY TEACUP AND SAUCER)</td>
      <td>(GREEN REGENCY TEACUP AND SAUCER)</td>
      <td>0.037658</td>
      <td>0.050032</td>
      <td>0.030909</td>
      <td>0.820768</td>
      <td>16.404818</td>
      <td>0.029024</td>
      <td>5.300218</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(GREEN REGENCY TEACUP AND SAUCER)</td>
      <td>(ROSES REGENCY TEACUP AND SAUCER )</td>
      <td>0.050032</td>
      <td>0.051264</td>
      <td>0.037551</td>
      <td>0.750535</td>
      <td>14.640537</td>
      <td>0.034986</td>
      <td>3.803087</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(ROSES REGENCY TEACUP AND SAUCER )</td>
      <td>(GREEN REGENCY TEACUP AND SAUCER)</td>
      <td>0.051264</td>
      <td>0.050032</td>
      <td>0.037551</td>
      <td>0.732497</td>
      <td>14.640537</td>
      <td>0.034986</td>
      <td>3.551247</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>(JUMBO  BAG BAROQUE BLACK WHITE)</td>
      <td>0.103814</td>
      <td>0.048747</td>
      <td>0.030534</td>
      <td>0.294118</td>
      <td>6.033613</td>
      <td>0.025473</td>
      <td>1.347609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(JUMBO  BAG BAROQUE BLACK WHITE)</td>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>0.048747</td>
      <td>0.103814</td>
      <td>0.030534</td>
      <td>0.626374</td>
      <td>6.033613</td>
      <td>0.025473</td>
      <td>2.398615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(JUMBO BAG PINK POLKADOT)</td>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>0.062085</td>
      <td>0.103814</td>
      <td>0.042051</td>
      <td>0.677308</td>
      <td>6.524245</td>
      <td>0.035605</td>
      <td>2.777218</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>(JUMBO BAG PINK POLKADOT)</td>
      <td>0.103814</td>
      <td>0.062085</td>
      <td>0.042051</td>
      <td>0.405057</td>
      <td>6.524245</td>
      <td>0.035605</td>
      <td>1.576478</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(JUMBO SHOPPER VINTAGE RED PAISLEY)</td>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>0.060692</td>
      <td>0.103814</td>
      <td>0.035194</td>
      <td>0.579876</td>
      <td>5.585724</td>
      <td>0.028893</td>
      <td>2.133149</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>(JUMBO SHOPPER VINTAGE RED PAISLEY)</td>
      <td>0.103814</td>
      <td>0.060692</td>
      <td>0.035194</td>
      <td>0.339009</td>
      <td>5.585724</td>
      <td>0.028893</td>
      <td>1.421061</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>(JUMBO STORAGE BAG SUKI)</td>
      <td>0.103814</td>
      <td>0.060531</td>
      <td>0.037390</td>
      <td>0.360165</td>
      <td>5.950055</td>
      <td>0.031106</td>
      <td>1.468299</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(JUMBO STORAGE BAG SUKI)</td>
      <td>(JUMBO BAG RED RETROSPOT)</td>
      <td>0.060531</td>
      <td>0.103814</td>
      <td>0.037390</td>
      <td>0.617699</td>
      <td>5.950055</td>
      <td>0.031106</td>
      <td>2.344190</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(LUNCH BAG RED RETROSPOT)</td>
      <td>(LUNCH BAG  BLACK SKULL.)</td>
      <td>0.074566</td>
      <td>0.065138</td>
      <td>0.032516</td>
      <td>0.436063</td>
      <td>6.694431</td>
      <td>0.027658</td>
      <td>1.657742</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(LUNCH BAG  BLACK SKULL.)</td>
      <td>(LUNCH BAG RED RETROSPOT)</td>
      <td>0.065138</td>
      <td>0.074566</td>
      <td>0.032516</td>
      <td>0.499178</td>
      <td>6.694431</td>
      <td>0.027658</td>
      <td>1.847829</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
