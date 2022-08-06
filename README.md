# Time Series Analysis
Analyzing Time Series of the Google Mobility Report of Mexico during COVID-19

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
## Correlation among the mobility indicators
To analyze the relationship between the six mobility indicators, we utilized `National Level` instances and performed a Pearson correlation. 
```
import seaborn as sn
corrMatrix = series.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
```

The image below shows that all variables are highly correlated. All of them are positively correlated, except for the `residential_percent_change_from_baseline`. variable. This negative relationship is logical since people during the pandemic would try to stay inside their homes while avoiding public places. Thus, residential mobility would have an increasing tendency, while the rest present a negative one.

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

