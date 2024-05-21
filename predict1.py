
import pandas as pd
import matplotlib.pyplot as plt


#reading the data


    
weather= pd.read_csv("weather.csv", index_col="DATE")

def fahrtocel(f):
    return (32*f-32)*(5/9)

#filtering the data for training the model

#cleaning missing values
null_pct=weather.apply(pd.isnull).sum()/weather.shape[0]
valid_columns=weather.columns [null_pct<0.04]
weather = weather[valid_columns].copy()
weather.columns=weather.columns.str.lower()
weather=weather.ffill()

#Convert Index into the datatype as date, remove gaps from data
weather.index=pd.to_datetime(weather.index)
weather.index.year.value_counts().sort_index()

   
# predicting max temp

weather["target"]=weather.shift(-1)["tmax"]
weather=weather.ffill() #what do you wanna predict for the next day

#apply ridge regression model
 
from sklearn.linear_model import Ridge

rr=Ridge(alpha=0.2) #initialising model

#list of columns we wish to predict

predictors=weather.columns[~weather.columns.isin(["target","name","station"])]


#backtesting_func tion

def backtest(weather, model, predictors, start=14260, step=100): 
    all_predictors=[]
    #loop: make predictions for 100 days and add it to the combine prediction to a huge db
    for i in range(start, weather.shape[0],step):
        train=weather.iloc[:i,:] # used to train the model
        test= weather.iloc[i:(i+step),:] #used to test it

        model.fit(train[predictors], train["target"]) #use predictors to train for the next day.. data of the past 40 years

        preds=model.predict(test[predictors]) 
        #prediction made!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        preds=pd.Series(preds, index=test.index)
        combined= pd.concat([test["target"],preds],axis=1)
        
        combined.columns = ["actual","prediction"]
        combined["diff"]=(combined["prediction"]-combined["actual"]).abs()
        
        all_predictors.append(combined) 
    return pd.concat(all_predictors)

    #return pd.concat

#predict

predictions= backtest(weather,rr,predictors)


#generate an accuracy metric : mean absolute error

from sklearn.metrics import mean_absolute_error

#mean_absolute_error(predictions["actual"],predictions["prediction"])
predictions["diff"].mean()

#improve the accuracy: calculate average prcp

def pct_diff(old, new):
    return (new-old)/old

def compute_rolling(weather, horizon, col): #optimising the prediction using the average from the horizon

    label=f"rolling_{horizon}_{col}"
    weather[label]=weather[col].rolling(horizon).mean()

    weather[f"{label}_pct"]=pct_diff(weather[label],weather[col])
    return weather

rolling_horizons=[3,10]

for horizon in rolling_horizons:
    for col in ["tmax","tmin","prcp"]:
        weather=compute_rolling(weather, horizon, col)

weather=weather.iloc[14:,:]
weather=weather.fillna(0)

#add more predictors

def expand_mean(df):  #takes in df and returns the mean of all the rows
    return df.expanding(1).mean()

for col in ["tmax","tmin","prcp"]:
    weather[f"month_avg_{col}"]= weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"]=weather[col].groupby(weather.index.day_of_year,group_keys=False).apply(expand_mean)




predictors=weather.columns[~weather.columns.isin(["target","name","station"])]
predictions= backtest(weather,rr,predictors)
predictions["diff"].mean()
predictions.sort_values("diff",ascending=False)

predictions['diff'].round().value_counts().sort_index().plot()
plt.title('Error Behaviour')
plt.show()


#predictions.to_csv('Predictions.csv',index=True)
print(predictions)

predictions.pop('diff')
predictions.plot (ylabel='temp max in celsius(predicted and actual)')
#predictions["diff"].round().value_counts().sort_index().plot()
plt.title('Comparative Predictions for Maximum Temperature')
plt.show()
