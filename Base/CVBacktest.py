import numpy as np
import pandas as pd




class CVBacktest(object):
    '''
    This is a class to run our backtesting strategy using a ML model object we designed. The strategy of the backtest is an
    iterative process that does the following
        1) split the data into a training set and test set based on a specified date
        2) using the training set, train your ML model to make forecasted returns
        3) based on forecasted returns rank the returns and create an equally-weighted long short portfolio
        4) find the selected stocks at each time index and generate the actually returns and performance metrics

    :param X : dataset containing our features (lagged returns for our basic case) ***this should be indexed on y months to avoiding given look ahead bias
    :param y : dataset containing our labels (unlagged returns for basic case)
    :param model : our ML object we created to run inside in the backtest object. all use python packages but created models
        to have the same fit and predict methods
    :param window(optional) : string to describe if the training dataset is a expanding window or a sliding window
    :param window_size(optional) : the size of the sliding window, expanding is always the entire set
    '''

    def __init__(self, X, y, model_list, start_date, end_date, window='expanding', window_size=120, update_model_period = 6, num_stocks=None):
        self.X = X
        self.y = y
        self.model_list = model_list
        self.model = model_list[0] # initiate a model that we can use to update our model later
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.window_size = window_size
        self.update_model_period = update_model_period
        if num_stocks == None:
            self.num_stocks = int(self.X.shape[1] / 5)
        else:
            self.num_stocks = num_stocks
        self.all_scores = []
        self.picked_mods = []

    def is_sklearn_mod(self):
        lm_mods = set([LinearRegression, Ridge, Lasso, ElasticNet, LassoLarsIC, DecisionTreeRegressor,
                       RandomForestRegressor])
        if self.model.__class__ in lm_mods:
            return True
        else:
            return False

    def split_data(self, date, x_y):
        '''
        This is a function to split the data from the date specified and turn theminto numpy arrays. The date specified will be stored as a test set
        and the data before this will be stored as the training set
        :param date (string) representing the specific
        '''
        if x_y == 'x':
            if self.window == 'expanding':
                train = self.X[self.X.index < date]
            else:
                train = self.X[self.X.index < date]
                train = train[-self.window_size:]

            test = self.X[self.X.index == date]
        elif x_y == 'y':
            if self.window == 'expanding':
                train = self.y[self.y.index < date]
            else:
                train = self.y[self.y.index < date]
                train = train[-self.window_size:]

            test = self.y[self.y.index == date]

        return train.to_numpy(), test.to_numpy()

    def cv_split(self, date):
        if self.window == 'expanding':
            #x_train and x_test dfs
            new_x = self.X[self.X.index < date]
            x_train = new_x.iloc[:-self.update_model_period, :]
            x_test = new_x.iloc[-self.update_model_period:,:]

            #only need y_train df
            new_y = self.y[self.y.index < date]
            y_train = new_y.iloc[:-self.update_model_period,:]
            y_test = new_y.iloc[-self.update_model_period:, :]
        else:
            #sliding window for X_train and x_test dfs
            new_x = self.X[self.X.index < date]
            x_train = new_x.iloc[-(self.window_size + self.update_model_period):-self.update_model_period,:]
            x_test = new_x.iloc[-self.update_model_period:, :]

            #sliding window for y_train df
            new_y = self.y[self.y.index < date]
            y_train = new_y.iloc[-(self.window_size + self.update_model_period):-self.update_model_period, :]
            y_test = new_y.iloc[-self.update_model_period:, :]

        return x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy()
        # return x_train, y_train, x_test, y_test


    def xval_score(self, preds, actual):
        # takes the average of the MSE and sMAPE. this seemed to be giving the best models for linear regressions
        # sMAPE
        preds = pd.DataFrame(preds, columns=self.X.columns)
        actual = pd.DataFrame(actual, columns=self.X.columns)
        scores = []
        for i in range(len(preds)):
            #rank the returns from largest to smallest
            sorted_preds = list(preds.iloc[i, :].nlargest(preds.shape[1]).index)
            sorted_actual = list(actual.iloc[i, :].nlargest(actual.shape[1]).index)
            #now find how many were acurately sorted
            correct_sorted = 0
            for j in range(len(sorted_preds)):
                if sorted_preds[j] == sorted_actual[j]:
                    correct_sorted += 1
            #take the percent correctly sorted
            scores.append(correct_sorted/len(sorted_preds))


        return np.mean(scores)



    def update_model(self, date):
        x_train, y_train, x_test, y_test = self.cv_split(date)
        scores = []
        for mod in self.model_list:
            mod.fit(x_train, y_train)
            preds = mod.predict(x_test)
            scores.append(self.xval_score(preds, y_test))
            del mod

        #use model with lowest xval score as the model for next period
        self.model = self.model_list[np.argmax(scores)]
        self.all_scores.append(scores)
        self.picked_mods.append(self.model)
        return self


    def month_forecast(self, date):
        '''
        This function is built to create the projected forecasts for a given date based on a ML model.
        :param date: the specific month
        :return: the forecast for each industry for that specific month
        '''

        # split training and test data
        x_train, x_test = self.split_data(date, 'x')
        y_train, __ytest = self.split_data(date, 'y')

        # make predictions for sklearn model
        if self.is_sklearn_mod() == True:
            mod = MultiOutputRegressor(self.model)
            mod.fit(x_train, y_train)
            pred = mod.predict(x_test)
            forecasts = list(pred.flatten())  # turn into list
        else:
            self.model.fit(x_train, y_train)
            pred = self.model.predict(x_test)
            forecasts = list(pred.flatten())

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

        # build model if it is a neural network
        if self.model.__class__ == NeuralNetwork:
            self.model.build_model(self.X.shape[1], self.y.shape[1])
        self.ind = 0
        for forecast_date in index:
            # create date in str(yyyy-mm) format so we can slice
            forecast_date = str(forecast_date.year) + '-' + str(forecast_date.month)
            #wether to update model using CV this month or not
            if self.ind % self.update_model_period == 0:
                self.update_model(forecast_date)

            month_returns = self.month_forecast(forecast_date)
            all_pred_returns.append(month_returns)
            self.ind += 1

        self.preds = pd.DataFrame(all_pred_returns, columns=cols, index=index)

        return self

    def make_portfolio(self):
        '''
        Function to take the forecasted returns and creates two dataframes, the one with long returns and one with short returns
        :return: long_returns, short_returns: dtaframes containing the actual returns of selcted stocks
        '''

        forecasted_returns = self.preds
        actual_returns = self.y[(self.y.index >= self.start_date) & (
                    self.y.index <= self.end_date)]  # this is where we get the actual returns from

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


    def eval_port_selection(self):
        '''
        Function to evaluate the stocks selected and show how accuarate the long and short bundles were at each position.
        This shows more rank accuracy of forecasts rather than
        :return: a dataframe containing 3 columns, top accuracy / short accuracy / total accuracy
        '''

        forecasted_returns = self.preds
        actual_returns = self.y[(self.y.index >= self.start_date) & (self.y.index <= self.end_date)]

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
        nyears = port_rets.shape[0] / 12

        # performance metrics
        # cumulative_rets = self.cumulative_returns()
        cumulative_retns = 100 * np.cumprod(1 + port_rets / 100)
        self.mean_monthly_return_annualized = np.mean(1 + port_rets / 100) ** 12 - 1

        # self.mean_return = (cumulative_rets.tail(1) / 100) ** (1.0 / nyears) - 1
        self.mean_return = (cumulative_retns[-1] / 100) ** (1.0 / nyears) - 1
        annualized_vol = np.std(port_rets / 100) * np.sqrt(12)
        self.sharpe = self.mean_monthly_return_annualized / annualized_vol

        return self



class CVBacktest(object):
    '''
    This is a class to run our backtesting strategy using a ML model object we designed. The strategy of the backtest is an
    iterative process that does the following
        1) split the data into a training set and test set based on a specified date
        2) using the training set, train your ML model to make forecasted returns
        3) based on forecasted returns rank the returns and create an equally-weighted long short portfolio
        4) find the selected stocks at each time index and generate the actually returns and performance metrics

    :param X : dataset containing our features (lagged returns for our basic case) ***this should be indexed on y months to avoiding given look ahead bias
    :param y : dataset containing our labels (unlagged returns for basic case)
    :param model : our ML object we created to run inside in the backtest object. all use python packages but created models
        to have the same fit and predict methods
    :param window(optional) : string to describe if the training dataset is a expanding window or a sliding window
    :param window_size(optional) : the size of the sliding window, expanding is always the entire set
    '''

    def __init__(self, Xs, y, model_list, start_date, end_date, window='expanding', window_size=120, update_model_period = 6, num_stocks=None):
        self.Xs = Xs
        self.X = None #placeholder for which X dataframe to use
        self.y = y
        self.model_list = model_list
        self.model = model_list[0] # initiate a model that we can use to update our model later
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.window_size = window_size
        self.update_model_period = update_model_period
        if num_stocks == None:
            self.num_stocks = int(self.Xs[0].shape[1] / 5)
        else:
            self.num_stocks = num_stocks
        for i in range(len(self.model_list)):
            if self.model_list[i].__class__ == NeuralNetwork:
                self.model_list[i].build_model(self.Xs[0].shape[1], 1)


    def split_data(self, date, i):
        '''
        This is a function to split the data from the date specified and turn theminto numpy arrays. The date specified will be stored as a test set
        and the data before this will be stored as the training set
        :param date (string) representing the specific
        '''
        self.X = self.Xs[i]  # select the X dataset
        if self.window == 'expanding':
            x_train = self.X[self.X.index < date]
            y_train = self.y[self.y.index < date]
            y_train = y_train.iloc[:, i]
        else:
            x_train = self.X[self.X.index < date]
            x_train = x_train.iloc[-self.window_size:]
            y_train = self.y[self.y.index < date]
            y_train = y_train.iloc[-self.window_size:, i]

        # test set
        x_test = self.X[self.X.index == date]

        x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
        y_train = y_train.to_numpy().reshape((-1, 1))

        return x_train, y_train, x_test

    def cv_split(self, date, i):
        self.X = self.Xs[i]
        if self.window == 'expanding':
            #x_train and x_test dfs
            new_x = self.X[self.X.index < date]
            x_train = new_x.iloc[:-self.update_model_period, :]
            x_test = new_x.iloc[-self.update_model_period:,:]

            #only need y_train df
            new_y = self.y[self.y.index < date]
            y_train = new_y.iloc[:-self.update_model_period,i]
            y_test = new_y.iloc[-self.update_model_period:, i]
        else:
            #sliding window for X_train and x_test dfs
            new_x = self.X[self.X.index < date]
            x_train = new_x.iloc[-(self.window_size + self.update_model_period):-self.update_model_period,:]
            x_test = new_x.iloc[-self.update_model_period:, :]

            #sliding window for y_train df
            new_y = self.y[self.y.index < date]
            y_train = new_y.iloc[-(self.window_size + self.update_model_period):-self.update_model_period, i]
            y_test = new_y.iloc[-self.update_model_period:, i]

        return x_train.to_numpy(), y_train.to_numpy().reshape((-1,1)), x_test.to_numpy(), y_test.to_numpy().reshape((-1,1))


    def xval_score(self, preds, actual):
        # takes the average of the MSE and sMAPE. this seemed to be giving the best models for linear regressions
        # sMAPE
        preds = pd.DataFrame(preds, index=self.y.columns).transpose()
        actual = pd.DataFrame(actual, index=self.y.columns).transpose()
        scores = []
        for i in range(len(preds)):
            #rank the returns from largest to smallest
            sorted_preds = list(preds.iloc[i, :].nlargest(preds.shape[1]).index)
            sorted_actual = list(actual.iloc[i, :].nlargest(actual.shape[1]).index)
            #now find how many were acurately sorted
            correct_sorted = 0
            for j in range(len(sorted_preds)):
                if sorted_preds[j] == sorted_actual[j]:
                    correct_sorted += 1
            #take the percent correctly sorted
            scores.append(correct_sorted/len(sorted_preds))

        return np.mean(scores)

    def update_model(self, date):

        scores = []
        for mod in self.model_list:
            preds, y_test = [], []
            for i in range(len(self.Xs)):
                x_train, y_train, x_test, y_test_i = self.cv_split(date, i)
                y_test.append(y_test_i.ravel().tolist())
                mod.fit(x_train, y_train)
                pred = mod.predict(x_test).ravel()
                preds.append(pred.tolist())
        scores.append(self.xval_score(preds, y_test))

        #use model with lowest xval score as the model for next period
        self.model = self.model_list[np.argmax(scores)]
        return self


    def month_forecast(self, date):
        '''
        This function is built to create the projected forecasts for a given date based on a ML model.
        :param date: the specific month
        :return: the forecast for each industry for that specific month
        '''

        forecasts = []
        for i in range(len(self.Xs)):
            x_train, y_train, x_test = self.split_data(date, i)
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

        self.ind = 0
        for forecast_date in index:
            # create date in str(yyyy-mm) format so we can slice
            forecast_date = str(forecast_date.year) + '-' + str(forecast_date.month)
            #wether to update model using CV this month or not
            if self.ind % self.update_model_period == 0:
                self.update_model(forecast_date)

            month_returns = self.month_forecast(forecast_date)
            all_pred_returns.append(month_returns)
            self.ind += 1

        self.preds = pd.DataFrame(all_pred_returns, columns=cols, index=index)

        return self

    def make_portfolio(self):
        '''
        Function to take the forecasted returns and creates two dataframes, the one with long returns and one with short returns
        :return: long_returns, short_returns: dtaframes containing the actual returns of selcted stocks
        '''

        forecasted_returns = self.preds
        actual_returns = self.y[(self.y.index >= self.start_date) & (
                    self.y.index <= self.end_date)]  # this is where we get the actual returns from

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


    def eval_port_selection(self):
        '''
        Function to evaluate the stocks selected and show how accuarate the long and short bundles were at each position.
        This shows more rank accuracy of forecasts rather than
        :return: a dataframe containing 3 columns, top accuracy / short accuracy / total accuracy
        '''

        forecasted_returns = self.preds
        actual_returns = self.y[(self.y.index >= self.start_date) & (self.y.index <= self.end_date)]

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
        nyears = port_rets.shape[0] / 12

        # performance metrics
        # cumulative_rets = self.cumulative_returns()
        cumulative_retns = 100 * np.cumprod(1 + port_rets / 100)
        self.mean_monthly_return_annualized = np.mean(1 + port_rets / 100) ** 12 - 1

        # self.mean_return = (cumulative_rets.tail(1) / 100) ** (1.0 / nyears) - 1
        self.mean_return = (cumulative_retns[-1] / 100) ** (1.0 / nyears) - 1
        annualized_vol = np.std(port_rets / 100) * np.sqrt(12)
        self.sharpe = self.mean_monthly_return_annualized / annualized_vol

        return self


























def make_all_combs(list1, list2):
    all_combs = [(i, j) for i in list1 for j in list2]
    # for i in list1:
        # for j in list2:
            # all_combs.append((i,j))
    return all_combs
alphas = [0.1, 0.3, 1, 3, 10]
l1_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
all_combs = make_all_combs(alphas, l1_ratios)

mods = []
for comb in all_combs:
    mods.append(ElasticNet(comb[0], comb[1]))
for alpha in alphas:
    mods.append(Lasso(alpha))
    mods.append(Ridge(alpha))

test_mods = []

for i in range(1,37):
    test_mods.append(CVBacktest(x, y, mods, '1970-01', '2017-01', update_model_period=i).all_forecasts())
test_mods_rets =[]
for m in test_mods:
    test_mods_rets.append(m.cumulative_returns())
plot_cumulative_returns(test_mods_rets, names)

names = [str(i) for i in range(1, 37)]
model = CVBacktest(x, y, [Naive()], '1970-01', '2017-01', update_model_period=36).all_forecasts()
model1 = CVBacktest(x,y, [LinearRegression()], '1970-01', '2017-01').all_forecasts()
plot_cumulative_returns([model.cumulative_returns(), model1.cumulative_returns()], ['Adapt', 'Static'])
model.port_performance()
model.sharpe
model1.port_performance()
model1.sharpe

sharpes = []
for m in test_mods:
    m.port_performance()
    sharpes.append(m.sharpe)

import pandas as pd
import numpy as np
d = pd.read_csv("C:\\Users\mozei\Downloads\MASE-EX1.xlsx")
def MASE(preds, actual):



def MASE(preds, y):
    numer = np.asarray(np.abs(preds - y))
    denom = np.asarray(np.sum(np.abs(y.diff()))/(y.shape[0]-1))
    mase = np.mean(np.sum(numer/denom)/y.shape[0])
    return mase