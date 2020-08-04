# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:21:25 2020

@author: gth667h
"""

def month_betas(self, date):
        betas = []
        for i in range(len(self.Xs)): #for each asset
            #beta_ind = []  #create a betas list
            x_train, y_train, x_test = self.split_data(date, i) #make training, testing
            if self.is_NN:
                pass
            else:
                mod = self.model.fit(x_train, y_train) #train the model
                bts = mod.coef_ #return the coeff
                #beta_ind.append(bts)
                #betas.append(beta_ind)
                betas.append(bts)
        return betas
    
    def all_betas(self):
        index = self.y.index[(self.y.index >= self.start_date) & (
                    self.y.index <= self.end_date)]  # list of dates we will be iterating through
        cols = self.y.columns  # list of column name for when we create the dateframe at the end
        all_betas = []  # this will store the returns -- will be turned into a dateframe

        #for Neural Networks
        if self.is_NN:
            self.build_NNs()
        #for classification approach
        if self.classif:
            self.y = self.make_classif_df(self.y)

        for forecast_date in index:
            # create date in str(yyyy-mm) format so we can slice
            forecast_date = str(forecast_date.year) + '-' + str(forecast_date.month)
            month_betas = self.month_betas(forecast_date)
            all_betas.append(month_betas)

        self.bets = pd.DataFrame(all_betas, columns=cols, index=index)
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
            if self.is_NN:
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

        progress_i = 0
        for forecast_date in index:
            # create date in str(yyyy-mm) format so we can slice
            forecast_date = str(forecast_date.year) + '-' + str(forecast_date.month)
            month_returns = self.month_forecast(forecast_date)
            all_pred_returns.append(month_returns)
            
            sys.stdout.write('.')
            progress_i += 1
            if progress_i % 80 == 0:
                print("")
                print("%s Still training step %d of %d" % (time.strftime("%H:%M:%S"), progress_i, len(index)))
            sys.stdout.flush()

        self.preds = pd.DataFrame(all_pred_returns, columns=cols, index=index)

        return self
