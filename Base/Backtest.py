from Base.Model_Class import *
import numpy as np
import pandas as pd
from scipy.stats import norm



class Backtest(object):
    '''
    This is a class to run our backtesting strategy using a ML model object we designed. The strategy of the backtest is an
    iterative process that does the following
        1) split the data into a training set and test set based on a specified date
        2) using the training set, train your ML model to make forecasted returns
        3) based on forecasted returns rank the returns and create an equally-weighted long short portfolio
        4) find the selected stocks at each time index and generate the actually returns and performance metrics

    :param X : a list of datasets containing our features
    :param y : dataset containing our labels (unlagged returns for basic case)
    :param model : our ML object we created to run inside in the backtest object. all use python packages but created models
        to have the same fit and predict methods
    :param window(optional) : string to describe if the training dataset is a expanding window or a sliding window
    :param window_size(optional) : the size of the sliding window, expanding is always the entire set
    '''

    def __init__(self, Xs, y, model, start_date, end_date, window='expanding', window_size=120, num_stocks=None, classif = False):
        self.Xs = Xs
        self.X = None #this is a placeholder for which ever X dataset we are using
        self.y = y
        self.actual = y #dummy variable made for holding the actual returns; in case calssifcation transforms y matrix
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.window_size = window_size
        self.classif = classif
        if num_stocks == None:
            self.num_stocks = int(self.y.shape[1] / 5)
        else:
            self.num_stocks = num_stocks
        if self.model.__class__ == NeuralNetwork:
            self.is_NN = True
        else:
            self.is_NN = False



    def build_NNs(self):
        if self.is_NN:
            mods = []
            i = 0
            while i < self.y.shape[1]:
                mods.append(self.model.build_model(self.Xs[0].shape[1], 1))
                i += 1
        self.model = mods
        return self

    def make_classif_df(self, df):
        '''Function that turns the y variable into a classification problem. This classification turns the
            returns into a 1 if above median for that month and a 0 otherwise'''
        df_copy = df.copy()
        for row in range(df.shape[0]):
            med = df.iloc[row, :].median()
            df_copy.iloc[row, :] = np.where(df.iloc[row, :] > med, 1, 0)

        return df_copy

    def split_data(self, date, i):
        '''
        This is a function to split the data from the date specified and turn theminto numpy arrays. The date specified will be stored as a test set
        and the data before this will be stored as the training set
        :param date (string) representing the specific
        '''

        self.X = self.Xs[i] #select the X dataset

        #this now can create the problem into a classification problem
        # if self.classif == True:
        #     y_train = self.make_classif_df(self.y)
        # else:
        #     y_train = self.y

        if self.window == 'expanding':
            x_train = self.X[self.X.index < date]
            y_train = self.y[self.y.index < date]
            y_train = y_train.iloc[:,i]
        else:
            x_train = self.X[self.X.index < date]
            x_train = x_train.iloc[-self.window_size:]
            y_train = self.y[self.y.index < date]
            y_train = y_train.iloc[-self.window_size:,i]

        #test set
        x_test = self.X[self.X.index == date]

        x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
        y_train = y_train.to_numpy()

        return x_train, y_train, x_test

    def month_forecast(self, date):
        '''
        This function is built to create the projected forecasts for a given date based on a ML model.
        :param date: the specific month
        :return: the forecast for each industry for that specific month
        '''

        forecasts = []

        for i in range(len(self.Xs)):
            x_train, y_train, x_test = self.split_data(date, i)
            if self.is_NN or type(self.model) == list:
                mod = self.model[i].fit(x_train, y_train)
            else:
                mod = self.model.fit(x_train, y_train)


            pred = mod.predict(x_test)
            forecasts.append(pred.item())

        return forecasts

    def all_forecasts(self):
        '''
        Function to preform the one month forecasts over all months which can be stored in a data frame for our
        portfolio strategy later
        '''

        index = self.y.index[(self.y.index >= self.start_date) & (
                    self.y.index <= self.end_date)]  # list of dates we will be iterating through
        cols = self.y.columns  # list of column name for when we create the dateframe at the end

        all_pred_returns = []  # this will store the returns -- will be turned into a dateframe

        #for Neural Networks
        if self.is_NN:
            self.build_NNs()
        #for classification approach
        if self.classif:
            self.y = self.make_classif_df(self.y)
        #for arima model
        if self.model.__class__ == AutoArima:
            self.model = []
            ts = self.y[:self.start_date]
            for i in range(self.y.shape[1]):
                # idk why mod.findorder(ts.iloc[:,i]) is return order (0,0,0) for all
                arima_model = AutoArima().find_order(ts.iloc[:, i])
                self.model.append(arima_model)

        for forecast_date in index:
            # create date in str format so we can slice
            forecast_date = str(forecast_date.year) + '-' + str(forecast_date.month) + '-' + str(forecast_date.day)
            month_returns= self.month_forecast(forecast_date)
            all_pred_returns.append(month_returns)


        self.preds = pd.DataFrame(all_pred_returns, columns=cols, index=index)

        return self

    def make_portfolio(self):
        '''
        Function to take the forecasted returns and creates two dataframes, the one with long returns and one with short returns
        :return: long_returns, short_returns: dtaframes containing the actual returns of selcted stocks
        '''

        forecasted_returns = self.preds
        actual_returns = self.actual[(self.actual.index >= self.start_date) & (
                    self.actual.index <= self.end_date)]  # this is where we get the actual returns from

        top_returns = []  # store the returns from the top stocks for each month
        bottom_returns = []  # store the returns from the top stocks for each month

        for i in range(len(forecasted_returns)):
            # find top/bottom n stocks from the forecasted returns
            top = list(forecasted_returns.iloc[i, :].nlargest(self.num_stocks).index)
            bottom = list(forecasted_returns.iloc[i, :].nsmallest(self.num_stocks).index)

            top_returns.append(list(actual_returns.iloc[i, :][top]))
            bottom_returns.append(list(actual_returns.iloc[i, :][bottom]))

        # turn the two list of lists into dataframes
        index = forecasted_returns.index
        top_returns = pd.DataFrame(top_returns, index=index)
        bottom_returns = pd.DataFrame(bottom_returns, index=index)

        return top_returns, bottom_returns

    def portfolio_names(self):
        '''Function to '''

    def MSE(self, by='mean'):
        '''
        Function to define the MSE of our forecasts.
        :param by ('mean', 'row', 'col') the way we will calculate the MSE. row wise will show the MSE for each industry.
                col wise will show the MSE for each month. mean will give a total MSE for the model
        :return: vector or scalar of the MSE
        '''
        forecasted_returns = self.preds
        actual_returns = self.actual[(self.actual.index >= self.start_date) & (
                    self.actual.index <= self.end_date)]  # this is where we get the actual returns from
        sq_resids = (forecasted_returns - actual_returns) ** 2
        if by == 'col':
            mse = sq_resids.sum(axis=1) / sq_resids.shape[1]
        elif by == 'row':
            mse = sq_resids.sum(axis=0) / sq_resids.shape[0]
        elif by == 'mean':
            mse = (sq_resids.sum(axis=0) / sq_resids.shape[0]).sum() / sq_resids.shape[1]

        return mse

    def sMAPE(self):
        # |(actual - forecasted)|/(|actual| + |forecasted|)
        val = np.abs((self.actual - self.preds))/(np.abs(self.actual)+np.abs(self.preds))
        # (2/forecast_horizon) * SUM(val)
        #sum top and take mean
        smape = (2/self.actual.shape[0])*np.sum(np.sum(val))
        return smape

    def MASE(self):
        numer = np.abs(self.preds - self.actual)
        denom = np.asarray(np.sum(np.abs(self.actual.diff()))/(self.actual.shape[0]-1))
        mase = np.mean(np.sum(numer/denom)/self.actual.shape[0])
        return mase

    def eval_port_selection(self):
        '''
        Function to evaluate the stocks selected and show how accuarate the long and short bundles were at each position.
        This shows more rank accuracy of forecasts rather than
        :return: a dataframe containing 3 columns, top accuracy / short accuracy / total accuracy
        '''

        forecasted_returns = self.preds
        actual_returns = self.actual[(self.actual.index >= self.start_date) & (self.actual.index <= self.end_date)]

        # store the percent of correctly selected
        top, bottom, total = np.empty(0), np.empty(0), np.empty(0)
        for i in range(len(forecasted_returns)):
            # find the percent of top stocks correctcly selected
            forecasted_top = list(forecasted_returns.iloc[i, :].nlargest(self.num_stocks).index)
            actual_top = list(actual_returns.iloc[i, :].nlargest(self.num_stocks).index)
            top_percent = len(set(forecasted_top).intersection(actual_top)) / self.num_stocks
            top = np.append(top, top_percent)

            # bottom
            forecasted_bottom = list(forecasted_returns.iloc[i, :].nsmallest(self.num_stocks).index)
            actual_bottom = list(actual_returns.iloc[i, :].nsmallest(self.num_stocks).index)
            bottom_percent = len(set(forecasted_bottom).intersection(actual_bottom)) / self.num_stocks
            bottom = np.append(bottom, bottom_percent)

        total = (top + bottom) / 2

        df = pd.DataFrame({'Top Percent': top,
                           'Bottom Percent': bottom,
                           'Total Percent': total},
                          index=forecasted_returns.index)

        return df

    def port_returns(self):
        '''
        Function to calculate the returns of our portfolio at each time step
        :return: vector containing the portfolios returns
        '''

        top_rets, bottom_rets = self.make_portfolio()
        # top_rets = test[0]
        # bottom_rets= test[1]
        port_rets = []
        port_rets.append(0)
        for return_month in range(1, top_rets.shape[0]):
            # return_month=month_index+1
            # take the mean of the long minus the mean of the short since everything is equally weighted
            port_ret = (top_rets.iloc[return_month - 1, :].mean()) / 2 - (
                bottom_rets.iloc[return_month - 1, :].mean()) / 2
            port_rets.append(port_ret)
            # port_rets.append(top_rets.iloc[i,:].mean() - bottom_rets.iloc[i,:].mean())
        port_rets = pd.DataFrame({'Returns': port_rets}, index=top_rets.index)

        return port_rets

    def cumulative_returns(self):
        # port_rets = self.port_returns().to_numpy()
        port_rets = self.port_returns()

        cumulative_rets = 100 * np.cumprod(1 + port_rets / 100)
        return cumulative_rets

    def port_performance(self):
        '''
        Function to calculate necessary portfolio evaluations
        :return:
        '''
        port_rets = self.port_returns().to_numpy()
        self.skew = pd.Series(port_rets.ravel()).skew()
        self.kurt = pd.Series(port_rets.ravel()).kurt()
        nyears = port_rets.shape[0] / 12

        # performance metrics
        # cumulative_rets = self.cumulative_returns()
        cumulative_retns = 100 * np.cumprod(1 + port_rets / 100)
        self.mean_monthly_return_annualized = np.mean(1 + port_rets / 100) ** 12 - 1

        # self.mean_return = (cumulative_rets.tail(1) / 100) ** (1.0 / nyears) - 1
        self.mean_return = (cumulative_retns[-1] / 100) ** (1.0 / nyears) - 1
        self.annualized_vol = np.std(port_rets / 100) * np.sqrt(12)
        self.sharpe = self.mean_monthly_return_annualized / self.annualized_vol
        self.VaR = norm.ppf(0.05, pd.Series(port_rets.ravel()).mean(), pd.Series(port_rets.ravel()).std())

        return self