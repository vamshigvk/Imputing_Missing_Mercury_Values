def calcMissing(readings):
    from datetime import datetime
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, SGDRegressor
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np

    dates = []
    temp_values = []
    for x in readings:
        temp_list = x.split('\t')
        float_days = datetime.strptime(temp_list[0], '%m/%d/%Y %H:%M:%S')
        dates.append(float_days)
        try:
            temp_values.append(float(temp_list[1]))
        except:
            temp_values.append(np.nan)
            pass

    temp_df = pd.Series(temp_values, index=dates)
    temp_df.index.name = 'Date'

    temp_df = temp_df.reset_index(name='Temp')
    missing_temp_dates = temp_df[temp_df['Temp'].isnull()]['Date'].values
    missing_temp_dates = missing_temp_dates.astype('datetime64[D]').astype(int)
    missing_temp_dates = [[x] for x in missing_temp_dates]
    missing_temp_dates = np.asarray(missing_temp_dates)

    temp_df = temp_df.dropna()
    dates, temps = [[x] for x in temp_df['Date'].values], temp_df['Temp'].values


    X,y = np.asarray(dates), np.asarray(temps)

    from sklearn.ensemble import GradientBoostingRegressor
    mdl = GradientBoostingRegressor()
    mdl.fit(X, y)

    y_pred = mdl.predict(missing_temp_dates)
    for pred in y_pred:
        print(pred)        

if __name__ == '__main__':
    readings_count = int(input().strip())

    readings = []

    for _ in range(readings_count):
        readings_item = input()
        readings.append(readings_item)

    calcMissing(readings)