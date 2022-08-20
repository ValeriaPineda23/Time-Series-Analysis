# Time Series Analysis
Analyzing Time Series of the Google Mobility Report of Mexico during COVID-19

Analyzed trends, seasonality, and stationarity behaviors from the Google Mobility Report of Mexico, during COVID-19. Using the information from the analysis, we developed an Auto-Regressive model that could predict Workplace mobility for a week of November 2020 with an RMSE of 6.32 for the training and 13.44 in the test set. This quantity means that, on average, the predictions differ from the actual value by only 13 units.

## Data Preprocessing
We performed data cleaning tasks initially, such as variable selection, data engineering, and handling missing data. 

### Variable Selection
Firstly, we eliminated the variables: `iso_3166_2_code`, `sub_region_2`, `metro_area`, `census_fips_code`, and `country_region`. The variable `iso_3166_2_code` was eliminated to avoid multicollinearity as it repeated information from variable `sub_region_1`. The variable `sub_region_2` was deleted as it did not add valuable information about the observations. Finally, the variables `metro_area`, `census_fips_code`, and `country_region` were composed entirely of missing values; thus, these were also removed.

```
series = pd.read_csv('2020_MX_Region_Mobility_Report.csv', header=0, index_col=0)
del series['sub_region_2']
del series['metro_area']
del series['census_fips_code']
del series['country_region']
```
### Data Engineering
Next, we set the `date` variable as a DateTime data type, and we proceeded to place it as the index of the data frame.

```
series['date'] = pd.to_datetime(series['date']).dt.strftime('%d/%m/%Y')
series.set_index('date', inplace=True)
```

### Missing Data
Moreover, we analyzed the missing data, which appeared in `sub_region_1` and `transit_stations_percent_change_from_baseline` variables. Variable `sub_region_1` was made up of 3.03% of missing data. However, according to the information given, missing data in `sub_region_1` represent data at a national level. Thus, missing values were imputed with the `National Level` category. 

```
series['sub_region_1'] = series['sub_region_1'].fillna('National Level')
```

On the other hand, `transit_stations_percent_change_from_baseline` variable presents 1.75% of missing data. Hence, we proceeded to treat them. As a time series analysis, we could not eliminate the instances that present missing data, so we decided to interpolate them.

```
series['transit_stations_percent_change_from_baseline'] = series['transit_stations_percent_change_from_baseline'].interpolate()
```

### Exploratory Data Analysis
#### Correlation among the mobility indicators
To analyze the relationship between the six mobility indicators, we utilized `National Level` instances and performed a Pearson correlation. 

```
import seaborn as sn
corrMatrix = series.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
```
The image below shows that all variables are highly correlated. All of them are positively correlated, except for the `residential_percent_change_from_baseline`. variable. This negative relationship is logical since people during the pandemic would try to stay inside their homes while avoiding public places. Thus, residential mobility would have an increasing tendency, while the rest present a negative one.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90649106/183266103-51389223-9890-4147-8d3c-d0fade62b23f.png" width="800" >
</p>
  
Furthermore, the image below shows that Retail, Parks, Workplaces and Transit, show a very similar tendency among them. The grocery and pharmacy sector show a slightly higher mobility, and the residential mobility shows an exactly opposing tendency to the rest. In addition, overall Pearson correlation is that of 0.85 among all indicators.
```
overall_pearson_r = series.corr().iloc[0,1]
r, p = stats.pearsonr(series.dropna()['retail_and_recreation_percent_change_from_baseline'], series.dropna()['grocery_and_pharmacy_percent_change_from_baseline'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")

# Compute rolling window synchrony
f,ax=plt.subplots(figsize=(10,3))
series.rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r')
ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}");
plt.legend(bbox_to_anchor = (0.8, -0.20))

plt.show()
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/90649106/183266109-ad9b46c6-0fec-4bee-9a19-6d1a48db0f54.png">
</p>
  
#### Analyze trend, seasonality, and stationarity
Next, we proceeded to analyze trend and seasonality of the six mobility indicators. 

```
data=series[series['sub_region_1']== 'National Level']

from statsmodels.tsa.seasonal import seasonal_decompose
for i in data.columns:
    if i != 'sub_region_1': 
        decomposition = seasonal_decompose(data[i], period=15)
        trend_estimate = decomposition.trend
        periodic_estimate = decomposition.seasonal
        residual = decomposition.resid
        print(f'############################# {i} ##############################')
        plt.subplot(221)
        plt.plot(data[i],label='Original time series', color='blue')
        plt.plot(trend_estimate ,label='Trend of time series' , color='red')
        plt.legend(loc='best',fontsize=8 , bbox_to_anchor=(0.90, -0.05))
        plt.subplot(222)
        plt.plot(trend_estimate,label='Trend of time series',color='blue')
        plt.legend(loc='best',fontsize=8, bbox_to_anchor=(0.90, -0.05))
        plt.subplot(223)
        plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
        plt.legend(loc='best',fontsize=8, bbox_to_anchor=(0.90, -0.05))
        plt.subplot(224)
        plt.plot(residual,label='Decomposition residuals of time series',color='blue')
        plt.legend(loc='best',fontsize=8, bbox_to_anchor=(1.09, -0.05))
        plt.tight_layout()
        
        plt.show()
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/90649106/183454178-9668596f-ddc3-4046-8265-07fce223f7de.png">
</p>

The graphic results show that all the indicators convey a particular trend in the first two months. For example, the Residential Mobility Indicator shows a steep increase and then begins to decrease and stabilize at the end. On the other hand, the rest of the indicators present a steep decrease in the first two months, and for the rest of the year, there appears to be a slightly increasing tendency. There also appears to exist a consistent seasonality every week and every two weeks.


To know if the time series is stationary or not, we plotted them with their corresponding rolling mean and standard deviation. In the graphs, we can see that the rolling mean is not sufficiently consistent, even if the Dickey-Fuller test says otherwise. 
  
```
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(14).mean() #pd.rolling_mean(timeseries, window=12)
    rolstd = timeseries.rolling(14).std() #pd.rolling_std(timeseries, window=12)
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
```
  
```
for i in data.columns:
    if i != 'sub_region_1':
        print(i)
        test_stationarity(data[i])
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/90649106/183440623-5def1f09-e5c0-45ac-914d-d8a89986b88d.png" width="700" >
</p>
  
So, we proceeded to subtract the shifted values from the current values. The result shows that all indicators present a more consistent rolling mean with this subtraction.

```
for i in data.columns:
    if i != 'sub_region_1':
        print(i)
        factor=data[i]-data[i].shift()
        test_stationarity(factor.dropna())
```

Moreover, to establish three periods that could benefit the accuracy of a prediction, we performed autocorrelation plots. These plots show that the periods of 7, 14, and 21 days have the highest autocorrelation.

```
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
```


```
for i in data.columns:
    if i != 'sub_region_1':
        lags = len(data[i])-1
        print(i)
        factor_diff = data[i] - data[i].shift()
        factor_diff.dropna(inplace=True)
        lag_acf = acf(factor_diff, nlags=30)
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(data[i])),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(data[i])),linestyle='--',color='gray')
        plt.title(f'Autocorrelation Function {i}')
        plt.show()
        
        k = abs(lag_acf).argsort()[::-1][:10]
        for i in k:
            print(f'{i}   {lag_acf[i]}')
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/90649106/183442731-3996c7f3-c6c5-4e8f-826c-6e141133024e.png" width="600" >
</p>

## Modeling

### Part 1 (MA, AR, ARIMA models)

Next, we built three models that could predict the Workplace indicator in a National Level.
For which we divided the dataset into training and test sets.

```
test_size = 7

train = work_data[:-test_size]
test = work_data[-test_size:]

train_diff = train - train.shift()
train_diff.dropna(inplace=True)

plt.plot(train, label='Training set')
plt.plot(test, label='Test set', color='orange')
plt.legend();
plt.show()
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/90649106/183442903-9f2dc9bb-5970-4afa-9b17-6954a60b8947.png" >
</p>

Next, we applied MA (Moving Average), AR (Auto-Regressive), and ARIMA (Auto-Regressive Integrated Moving Average) models. These were computed using the ARIMA function from the *statsmodels.tsa.arima_model* library in Python. The parameters established ( $p, d, q$ ) and the results for each model appear in the next table and graphs. 

Model|Parameters (p,d,q)|Training set (RMSE)|Test set (RMSE)
--- | --- | --- | --- 
MA|0,1,2|11.9121|20.6715
AR|7,1,0|6.3204|13.0160
ARIMA|7,1,2|6.0772|13.4449

<p align="center">
  <img src="https://user-images.githubusercontent.com/90649106/183446734-b453681e-0b83-4dd1-9385-90c0335aec94.png" >
</p>

From these results, we can see that the AR model has the lowest RMSE for the test set, which means that it is the best model to make predictions. The parameters of the AR model signify that to make good predictions, we need to perform a 1 step transformation to control more the non-stationarity, and we only need the information from 7 days before.

The predictions for the test set using AR model are the following:


Day|Real value|Prediction
--- | --- | --- 
11/11/2020|-29|-29.2514
12/11/2020|-29|-29.7783
13/11/2020|-24|-25.1375
14/11/2020|-3|-7.1343
15/11/2020|5|-1.3488
16/11/2020|-54|-21.9345
17/11/2020|-29|-42.3344


The predictions using AR presented a RMSE of 13.44 for the test set. This quantity means that, on average, the predictions differ from the actual value by 13 units. However, we can see that the error is biased by the predictions of 16/11/2020. This date appears to be an outlier, as it does not follow a typical pattern with respect to the rest of the time series.

# Conclusion
The correlation analysis between the mobility indicators allowed us to know if there exists a dependency among them. In this case, we can conclude that if people have an increasing presence in their homes, they will not have a high presence in other areas.

Moreover, we visualized the trend and seasonality of each indicator, from wich we conclude that there exists a highly negative trend for the first two months in almost all indicators (except for the Residential mobility, which shows a positive trend). In addition, there is a seasonality of 7, 14, and 21 days, which means that the pattern repeats in a weekly form. Finally, we subtracted a time shift from the current time to avoid a non-stationary behavior.

Using the previous information, we developed an Auto-Regressive model that could predict Workplace mobility for the last seven days with an RMSE of 6.32 for the training set and 13.44 in the test set.
