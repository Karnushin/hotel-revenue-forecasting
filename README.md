I use notebooks and no hard repository structure for simplicity here

**Data**

data includes booking transactions from 2 different hotels located in 2 different locations

**Goal**

Predict the next rolling 6-month revenue for each hotel.

**Python**

Version 3.7 is used

**Libraries**

Some standard and libraries from requirements.txt

**Files**

 - 1_simple_data_change.ipynb - creating full arrival_date and year_month date
 - 2_EDA.ipynd - big EDA on the dataset with comments
 - 3_modeling.ipynb - building ML models, metrics, different comments
 - utils.py - auxiliary functions used in EDA and modeling
 - requirements.txt - requirements file with libraries versions

**Part of useful and interesting articles**
 
 - https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=9026573&fileOId=9026574 - conf int LSTM
 - https://towardsdatascience.com/time-series-forecasting-prediction-intervals-360b1bf4b085#:~:text=This%20is%20where%20prediction%20intervals,fall%20within%20the%20prediction%20interval - ARIMA + CI
 - https://www.cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning.html - Prediction intervals when forecasting with machine learning models
