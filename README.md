# Liquor_Store_CaseStudy
A small case study, where I used Python to analyze liquor store sales data, in order to extract some useful insights.

## Table of Contents

-[Project Overview](project-overview)

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

1-As a first step, we isolated and extracted only the sales that were made during the period from 2016 to 2019. We did it using the following code:

```python
#transform the date column from str to date format
df['date']= pd.to_datetime(df['date'])

#include only the years between 2016 and 2019
df = df.loc[(df['date'] >= '2016-01-01')
                     & (df['date'] < '2020-01-01')]
```

2-Now, our updated main dataset (df) contains sales that were made during the period from 2016 to 2019. As a next step, we created two separate subsets to use for the two assignment tasks.

   In order to create the first subset (df_1), we used the following code, in order to choose the relevant columns from the main dataset (df):

 ```python
 # Firstly, we choose the relevant columns for the core dataset
 df_1 = df[['zip_code','item_number','bottles_sold']]
 ```
    
 After confirming that the columns had the correct format, we used the following code to find and delete any rows that may contain missing values. 
 
 ```python 
#then we delete any row with missing values and reset the index 
df_1 = df_1.dropna()
df_1.reset_index(inplace = True)
del(df_1['index'])
```

We also followed the exactly same two steps, in order to create the subset used for the second task (df_2). We used the following code to choose the relevant columns for the main dataset:

```python
# Firstly, we choose the relevant columns for the core dataset
df_2 = df[['store_name','sale_dollars']]
```

And we deleted any rows that contained missing values:

``` python
#then we delete any row with missing values and reset the index 
df_2 = df_2.dropna()
df_2.reset_index(inplace = True)
del(df_2['index'])
df_2
```

The two subsets have the following form

- df_1:

|index|zip_code|item_number|bottles_sold|
|-----|--------|-----------|------------|
|0    |52556   |86251      |6           |
|1    |52803   |41844      |3           |
|2    |51501   |81124      |2           |
|3    |52761   |82847      |4           |
|4    |50320   |973627     |120         |
|5    |50158   |67557      |6           |
|6    |50703   |168        |180         |
|7    |51247   |86112      |6           |
|8    |50702   |100026     |84          |
|9    |50501   |38176      |108         |
|10   |50461   |35918      |30          |
|11   |50588   |84617      |4           |
|12   |52241   |43034      |1           |
|13   |51501   |86251      |48          |
|14   |50314   |48106      |60          |
|15   |52601   |48099      |24          |
|16   |50314   |49182      |24          |
|17   |50707   |15626      |60          |
|18   |51401   |33717      |4           |
|19   |50158   |67528      |3           |
|20   |52314   |75087      |660         |
|21   |50320   |926577     |66          |
|22   |50158   |48099      |24          |
|23   |50314   |905153     |36          |
|24   |51501   |67524      |7           |
|25   |52314   |75087      |900         |
|26   |51534   |84197      |3           |
|27   |52411   |41846      |36          |
|28   |52804   |48690      |2           |
|29   |51106   |86251      |144         |
|30   |52240   |27102      |18          |
|31   |50401   |986845     |48          |
|32   |50314   |56193      |24          |
|33   |50010   |946574     |288         |
|34   |52601   |35913      |48          |
|35   |51106   |67586      |84          |
|36   |51360   |77487      |48          |
|37   |50263   |3135       |84          |
|38   |50266   |250        |90          |
|39   |50314   |86251      |180         |
|40   |52405   |84197      |2           |
|41   |50701   |43031      |4           |
|42   |52003   |43031      |5           |
|43   |50131   |38089      |48          |
|44   |50265   |67526      |72          |
|45   |50401   |86843      |18          |
|46   |51246   |82187      |5           |
|47   |50662   |86739      |8           |
|48   |52172   |67524      |1           |
|49   |50320   |77487      |72          |
|50   |52732   |45248      |24          |
|51   |50701   |43034      |7           |
|52   |50703   |43034      |1           |
|53   |50327   |43040      |102         |
|54   |51401   |27189      |18          |
|55   |50317   |56193      |24          |
|56   |52338   |27357      |90          |
|57   |52240   |86251      |60          |
|58   |50022   |86507      |4           |
|59   |50702   |917914     |18          |
|60   |51106   |67527      |240         |
|61   |50703   |48099      |24          |
|62   |50801   |43037      |5           |
|63   |51555   |45247      |2           |
|64   |51401   |43031      |12          |
|65   |50703   |37880      |24          |
|66   |52627   |67586      |36          |
|67   |51501   |43031      |4           |
|68   |50111   |77805      |108         |
|69   |50316   |48099      |48          |
|70   |50702   |77487      |768         |
|71   |50701   |86112      |2           |
|72   |52402   |45246      |7           |
|73   |52136   |35917      |2           |
|74   |52241   |65750      |48          |
|75   |52001   |67557      |4           |
|76   |52402   |86390      |216         |
|77   |50314   |86251      |60          |

- df_2:

|index|store_name                                |sale_dollars|
|-----|------------------------------------------|------------|
|0    |Hy-Vee Food Store / Fairfield             |21.78       |
|1    |Hilltop Grocery                           |18.9        |
|2    |Hy-Vee Drugstore / Council Bluffs         |17.82       |
|3    |Nash Finch / Wholesale Food               |45.72       |
|4    |Hy-Vee #3 / BDI / Des Moines              |1755.6      |
|5    |Hy-Vee Food Store / Marshalltown          |75.54       |
|6    |Hy-Vee Food Store #2 / Waterloo           |1985.4      |
|7    |Pump N Pak                                |21          |
|8    |Hy-Vee Food Store #3 / Waterloo           |1448.16     |
|9    |Hy-Vee Food Store / Fort Dodge            |1563.84     |
|10   |Osage Payless Foods                       |324         |
|11   |Hy-Vee Wine and Spirits / Storm Lake      |33          |
|12   |Bootleggin' Barzini's Fin                 |6.75        |
|13   |I-80 Liquor / Council Bluffs              |174.24      |
|14   |Central City 2                            |1709.4      |
|15   |Burlington Shell                          |206.64      |
|16   |Shop N Save #1 / Mlk Pkwy                 |116.64      |
|17   |Fareway Stores #067 / Evansdale           |1349.4      |
|18   |Fareway Stores #409 / Carroll             |32.52       |
|19   |Depot Liquor & Grocery                    |112.47      |
|20   |Wilkie Liquors                            |4870.8      |
|21   |Hy-Vee #3 / BDI / Des Moines              |1881        |
|22   |East Side Liquor and Groceries            |206.64      |
|23   |Central City Liquor, Inc.                 |1078.92     |
|24   |Speedy Gas N Shop                         |68.18       |
|25   |Wilkie Liquors                            |6750        |
|26   |Gameday Liquor                            |27          |
|27   |Hy-Vee #7 / Cedar Rapids                  |486         |
|28   |Famous Liquors                            |17.98       |
|29   |Sam's Club 6432 / Sioux City              |522.72      |
|30   |Hy-Vee Wine and Spirits / Iowa City       |488.52      |
|31   |Hy-Vee Food Store #1 / Mason City         |630.24      |
|32   |Shop N Save #1 / Mlk Pkwy                 |70.56       |
|33   |Sam's Club 6568 / Ames                    |3913.92     |
|34   |Quik Stop  /  Burlington                  |81.6        |
|35   |Sam's Club 6432 / Sioux City              |720.72      |
|36   |Hy-Vee Wine and Spirits / Spirit Lake     |324.96      |
|37   |Hy-Vee / Waukee                           |1518.72     |
|38   |Hy-Vee Wine and Spirits / WDM             |3372.3      |
|39   |Central City 2                            |653.4       |
|40   |Smokin' Joe's #7 Tobacco and Liquor Outlet|18          |
|41   |New Star Liquor / W 4th S / Waterloo      |33.24       |
|42   |Sid's Beverage Shop                       |41.55       |
|43   |Hy-Vee Food Store / Johnston              |64.32       |
|44   |Fareway Stores #153  /  W Des Moines      |1349.28     |
|45   |Hy-Vee Food Store #1 / Mason City         |270         |
|46   |Liquor Locker                             |37.5        |
|47   |Fareway Stores #412 / Oelwein             |105.04      |
|48   |Double D Liquor Store                     |9.74        |
|49   |Hy-Vee #3 / BDI / Des Moines              |487.44      |
|50   |Hy-Vee Food and Drug / Clinton            |282.24      |
|51   |New Star Liquor / W 4th S / Waterloo      |47.25       |
|52   |Hy-Vee Food Store #2 / Waterloo           |6.75        |
|53   |Fareway Stores #138 / Pleasant Hill       |2295        |
|54   |Hy-Vee Food Store / Carroll               |637.74      |
|55   |Hy-Vee Wine and Spirits / Hubbell         |70.56       |
|56   |Cedar Ridge Vineyards                     |3712.5      |
|57   |Hy-Vee Wine and Spirits / Iowa City       |217.8       |
|58   |Hy-Vee Wine and Spirits / Atlantic        |23.04       |
|59   |Hy-Vee Food Store #3 / Waterloo           |807.84      |
|60   |Sam's Club 6432 / Sioux City              |5397.6      |
|61   |Ray's Supermarket, Inc.                   |206.64      |
|62   |Fareway Stores #597 / Creston             |71.25       |
|63   |Food Land Super Markets                   |13.26       |
|64   |Hy-Vee Food Store / Carroll               |99.72       |
|65   |Logan Convenience Store                   |63.36       |
|66   |Quicker Liquor Store                      |237.24      |
|67   |Tobacco Hut #14 / Council Bluffs          |33.24       |
|68   |Fareway Stores #983 / Grimes              |1296        |
|69   |Tequila's Liquor Store                    |413.28      |
|70   |Sam's Club 6514 / Waterloo                |5199.36     |
|71   |Hy-Vee Wine and Spirits / Waterloo        |7           |
|72   |CVS Pharmacy #8526 / Cedar Rapids         |42          |
|73   |The Ox & Wren Spirits and Gifts           |75.12       |
|74   |Hy-Vee Food Store / Coralville            |504         |
|75   |Iowa Street Market, Inc.                  |50.36       |
|76   |Sam's Club 8162 / Cedar Rapids            |691.2       |
|77   |Central City 2                            |217.8       |

### Task 1

Our first task was to discern the most popular item in each zip code. We used the following code to create the "bottles_per_zipcode" table, in order to find the most popular item for each zip code:

```python
#we group the dataset by the zip code and sum the sold bottles
bottles_per_zipcode=df_1.groupby(['zip_code','item_number'],as_index = False )['bottles_sold'].sum()
```

The code above produced the following table:

|index|zip_code|item_number|bottles_sold|
|-----|--------|-----------|------------|
|0    |50010   |946574     |288         |
|1    |50022   |86507      |4           |
|2    |50111   |77805      |108         |
|3    |50131   |38089      |48          |
|4    |50158   |48099      |24          |
|5    |50158   |67528      |3           |
|6    |50158   |67557      |6           |
|7    |50263   |3135       |84          |
|8    |50265   |67526      |72          |
|9    |50266   |250        |90          |
|10   |50314   |48106      |60          |
|11   |50314   |49182      |24          |
|12   |50314   |56193      |24          |
|13   |50314   |86251      |240         |
|14   |50314   |905153     |36          |
|15   |50316   |48099      |48          |
|16   |50317   |56193      |24          |
|17   |50320   |77487      |72          |
|18   |50320   |926577     |66          |
|19   |50320   |973627     |120         |
|20   |50327   |43040      |102         |
|21   |50401   |86843      |18          |
|22   |50401   |986845     |48          |
|23   |50461   |35918      |30          |
|24   |50501   |38176      |108         |
|25   |50588   |84617      |4           |
|26   |50662   |86739      |8           |
|27   |50701   |43031      |4           |
|28   |50701   |43034      |7           |
|29   |50701   |86112      |2           |
|30   |50702   |77487      |768         |
|31   |50702   |100026     |84          |
|32   |50702   |917914     |18          |
|33   |50703   |168        |180         |
|34   |50703   |37880      |24          |
|35   |50703   |43034      |1           |
|36   |50703   |48099      |24          |
|37   |50707   |15626      |60          |
|38   |50801   |43037      |5           |
|39   |51106   |67527      |240         |
|40   |51106   |67586      |84          |
|41   |51106   |86251      |144         |
|42   |51246   |82187      |5           |
|43   |51247   |86112      |6           |
|44   |51360   |77487      |48          |
|45   |51401   |27189      |18          |
|46   |51401   |33717      |4           |
|47   |51401   |43031      |12          |
|48   |51501   |43031      |4           |
|49   |51501   |67524      |7           |
|50   |51501   |81124      |2           |
|51   |51501   |86251      |48          |
|52   |51534   |84197      |3           |
|53   |51555   |45247      |2           |
|54   |52001   |67557      |4           |
|55   |52003   |43031      |5           |
|56   |52136   |35917      |2           |
|57   |52172   |67524      |1           |
|58   |52240   |27102      |18          |
|59   |52240   |86251      |60          |
|60   |52241   |43034      |1           |
|61   |52241   |65750      |48          |
|62   |52314   |75087      |1560        |
|63   |52338   |27357      |90          |
|64   |52402   |45246      |7           |
|65   |52402   |86390      |216         |
|66   |52405   |84197      |2           |
|67   |52411   |41846      |36          |
|68   |52556   |86251      |6           |
|69   |52601   |35913      |48          |
|70   |52601   |48099      |24          |
|71   |52627   |67586      |36          |
|72   |52732   |45248      |24          |
|73   |52761   |82847      |4           |
|74   |52803   |41844      |3           |
|75   |52804   |48690      |2           |

Then, we used plotly to visualize our results and make them comprehensible. Our visualization code is the following:

``` python
#plotly graph
fig_1 = px.scatter(bottles_per_zipcode,x = 'zip_code', y= 'bottles_sold', size = 'bottles_sold',
                   color = 'zip_code', hover_name = bottles_per_zipcode.index, size_max=40,
                  labels = {'bottles_sold':'Bottles Sold',
                             'zip_code':'Zip Code'},title = 'Bottles Sold per Zip Code')
```

![image](https://github.com/DimitriosLavd/Liquor_Store_CaseStudy/assets/157892523/d8fa1e20-5b49-40e7-a348-a280cab4a357)
### Task 2

For our second task, we needed to compute the sales percentage per store (in dollars). We used the following code to create the "top_sales" table. We made sure that the table contained the top 16 selling stores and the sales percentage (in dollars) made by each store. We followed the following procedure and wrote the code below to create the table "top_sales" table:

```python
#we group the dataset by the store name and sum the sales in dollars
sales_percentage = df_2.groupby(['store_name'],as_index = False)['sale_dollars'].sum()
#we calculate the total sales in dollars and add the perncentage column to our dataset 
total_sales = sum(sales_percentage['sale_dollars'])
#we create the percentage column
sales_percentage['percentage'] = (sales_percentage['sale_dollars']/total_sales)*100
#we make sure that the results appear in an descenting order 
sales_percentage = sales_percentage.sort_values(by=['percentage'], ascending=False)
sales_percentage.reset_index(inplace = True)
del(sales_percentage['index'])
#we find the 15 top selling stores
top_sales = sales_percentage[:16]
```

We again used plotly to visualize our results with the following code:

``` python
fig_2 = px.bar(top_sales[::-1], x='percentage', y='store_name', color='percentage', orientation='h',text_auto='.2s',
              labels = {'store_name':'Store Name',
                       'percentage':'%Sales'},title = '%Sales by Store')
fig_2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
```


![image](https://github.com/DimitriosLavd/Liquor_Store_CaseStudy/assets/157892523/1746d255-0a40-44e8-8f83-06ff4ff07099)

### Insights 

The previous analysis and the graphs we produced can give us valuable insights, regarding the liquor sales, for the 2016–2019 period. In this section, we will give some examples.
- As we can see in the first graph, the most, bottles of a single item that were sold to a single zip code were 1560. The item was the 75087, and it was sold to the area with the zip code 52314
- In the second graph, we can see the top-selling stores. The top seller, in terms of dollars, was the Wilkie Liquors store with the 18 % of the total sales.
- The 16 top-selling stores, alone, made the 85% of the total sales in dollars from 2016 to 2019
  These are only some key points we can extract from the analysis we did. Further exploring the dataset and the graphs can provide additional information about the liquor sales for the targeted time 
   period 











