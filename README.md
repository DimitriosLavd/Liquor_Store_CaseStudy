# Liquor_Store_CaseStudy
A small case study, where I used Python to analyze liquor store sales data, in order to extract some useful insights.

### Project Overview

This project is an assignment, that is included in the Machine Learning and AI Bootcamp by Workearly. During this assignment, we were tasked to analyze a data set that contained liquor stores' sales data, using Python. The objective of our analysis was to extract useful insights regarding the top-selling areas and the top-selling stores for a specific time period. Most accurately, the tasks at hand were the following: 

For the timeframe 2016-2019:
1. Discern the most popular item in each zip code 
2. Compute the sales percentage per store (in dollars).

### Data Sources

Liquor sales data: The primary dataset used for this analysis is the "finance_liquor_sales.csv", containing detailed information about each sale.  This data set is provided by the Workearly's Machine Learning and AI Bootcamp. We imported the raw dataset as shown below:

```python
df = pd.read_csv(r"D:\data analysis_2\Case Studies\Liquor_sales\finance_liquor_sales.csv")
```

### Tools

The whole analysis was executed using Python and an array of relevant Python Libraries, such as:

- Pandas
- Numpy
- Plotly

We imported the relevant libraries as shown in the code below: 

```python
import pandas as pd
import numpy as np
import plotly.express as px
```

### Data Cleaning/Preperation

During the first stage of our analysis, we needed to clean and prepare the raw dataset, in order to make it useful for the tasks at hand.  

1. As a first step, we isolated and extracted only the sales that were made during the period from 2016 to 2019. We did it using the following code:

```python
#transform the date column from str to date format
df['date']= pd.to_datetime(df['date'])

#include only the years between 2016 and 2019
df = df.loc[(df['date'] >= '2016-01-01')
                     & (df['date'] < '2020-01-01')]
```






