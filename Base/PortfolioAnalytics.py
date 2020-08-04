
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_cumulative_returns(list_of_returns, labels):
    '''
    Function to plot the cumulative returns of a model.
    :param list_of_returns: THIS NEEDS TO BE A LIST [model1, model2, ...] each element in list is a pd.DataFrame of cumulative returns
    :param labels: a list containing what label names for each line on graph
    :return: a plot
    '''
    x_ax = list_of_returns[0].index
    for i in range(len(list_of_returns)):
        plt.plot(x_ax, list_of_returns[i].values, label=labels[i])
    plt.ylabel('Portfolio Returns ($)')
    plt.xlabel('Date')
    plt.title('Cumulative Portfolio Returns')
    plt.legend(loc='best')
    plt.show()


def plot_port_selection(eval_df):
    '''
    Function to plot how well the portfolio selected the correct top and bottom preformers
    :param returns: eval_df containg the portfolio selection statistics
    :return: plot
    '''
    x_ax = eval_df.index
    plt.subplot(311)
    plt.plot(x_ax, eval_df.iloc[:,0].values)
    plt.ylabel('Top Selection')
    plt.title('Percent of Corrected Predicted Top/Bottom Preformers')
    plt.ylim([0,1])

    plt.subplot(312)
    plt.plot(x_ax, eval_df.iloc[:,1].values)
    plt.ylabel('Bottom Selection')
    plt.ylim([0, 1])

    plt.subplot(313)
    plt.plot(x_ax, eval_df.iloc[:,2].values)
    plt.ylim([0, 1])
    plt.ylabel('Total Selection')
    plt.xlabel('Date')

    plt.tight_layout()
    plt.show()


def model_statistics(list_of_models):
    '''
    Function to calculate MSE, SHARPE and RETURN for several models.
    :param list_of_models: list of models already built
    :return: DF of metrics
    '''
    est_names, mse_list, smape_list, sharpe_list, returns_list = [], [], [], [], []

    for model in list_of_models:
        model.port_performance()  # doesnt return anything, but cant comment out or wont return sharpe, mean return
        est_names.append(type(model.model).__name__)
        mse_list.append(model.MSE())
        smape_list.append(model.sMAPE())
        sharpe_list.append(model.sharpe)
        returns_list.append(model.mean_return)

    skmetricsframe = pd.DataFrame(list(zip(est_names, mse_list, smape_list, sharpe_list, returns_list)),
                                  columns=['name', 'mse', 'mape', 'sharpe', 'returns'])

    return skmetricsframe


def naive_scaled_errors(model, naive_errors):
    '''Function to scale the forecasting errors in terms of a naive benchmark approach. As a validity check,
    if you run a naive model all your errors should come out to 1
    :param (model) either a model or a list of models to get scaled errors for
    :param (naive_errors) either a backtest object or a list of the naive errors for the first MSE, sMAPE, MASE
    :returns (errors_df) a dataframe of the scaled errors
    '''
    error_df = []
    #check if naive errors is list of backtest object
    if naive_errors != list:
        naive_errors = [naive_errors.MSE(), naive_errors.sMAPE(), naive_errors.MASE()]
    #check if model is one or a list of models
    if type(model) == list:
        for mod in model:
            errors = [type(mod.model).__name__, mod.MSE(), mod.sMAPE(), mod.MASE()]
            #scale the errors
            for i in range(1, len(errors)):
                errors[i] = errors[i]/naive_errors[i-1]
            #add the OWA error
            errors.append(sum(errors[1:])/len(naive_errors))
            error_df.append(errors)
    else:
        errors = [type(model.model).__name__, model.MSE(), model.sMAPE(), model.MASE()]
        for i in range(1, len(errors)):
            errors[i] = errors[i] / naive_errors[i-1]
        errors.append(sum(errors[1:])/len(naive_errors))
        error_df.append(errors)
    cols = ['Model', 'MSE/Naive', 'sMAPE/Naive', 'MASE/Naive', 'OWA/Naive']
    error_df = pd.DataFrame(error_df, columns=cols)

    return error_df





