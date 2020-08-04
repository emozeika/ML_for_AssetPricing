

from Base.preprocessing_data import *
from Base.Model_Class import *
from Base.Backtest import *
from Base.EnsembleBacktest import *
from Base.PortfolioAnalytics import *
import pickle


#load industry data
industry = read_ts_csv('Data/30industryreturns.csv', format='%Y%m')
rf = read_ts_csv('Data/rf.csv', format='%Y%m')
mkt = read_ts_csv('Data/mkt.csv', format='%Y%m')
mkt_st = read_ts_csv('Data/mkt_st.csv', format='%Y%m')

#this will be the start and end dates for all preprocessing
start, end = '1980-02', '2018-01'

#create the lag returns
ind_x, ind_y = lag_returns(industry)

#make dfs with momentum, alphas, betas
all_ind_x2 = get_all_predictors(ind_x[start:end], mkt[start:end], mkt_st[start:end])
ind_y = ind_y[all_ind_x2[0].index[0]:end]

#make the list of x matricies of just lagged returns for same time period
all_ind_x = [ind_x[all_ind_x2[0].index[0]:end] for i in range(ind_x.shape[1])]

#saving the data
pickle.dump(all_ind_x, open('Data/ind_x.sav', 'wb'))
pickle.dump(all_ind_x2, open('Data/ind_x_all_vars.sav', 'wb'))
pickle.dump(ind_y, open('Data/ind_y.sav', 'wb'))

#STATISTICAL BENCHMARKS
#these are the benchmark for all forecasting errors
#we only need to run this once because it doesnt use predictors
naive_mod = Backtest(all_ind_x, ind_y, Naive(), '2010-01', '2018-01').all_forecasts()
drift_mod = Backtest(all_ind_x, ind_y, Drift(), '2010-01', '2018-01').all_forecasts()
pm_mod = Backtest(all_ind_x, ind_y, PrevailingMean(), '2010-01', '2018-01').all_forecasts()
arima_mod = Backtest(all_ind_x, ind_y, AutoArima(), '2010-01', '2018-01').all_forecasts()

pickle.dump(naive_mod, open('Models/StatModels/naive_ind.sav', 'wb'))
pickle.dump(drift_mod, open('Models/StatModels/drift_ind.sav', 'wb'))
pickle.dump(pm_mod, open('Models/StatModels/pm_ind.sav', 'wb'))
pickle.dump(arima_mod, open('Models/StatModels/arima_ind.sav', 'wb'))

model_statistics([naive_mod, drift_mod, pm_mod, arima_mod])

#LINEAR REGRESSION
lm_mod = Backtest(all_ind_x, ind_y, LinearRegression(), '2010-01', '2018-01').all_forecasts()
lm_mod1 = Backtest(all_ind_x2, ind_y, LinearRegression(), '2010-01', '2018-01').all_forecasts()

pickle.dump(lm_mod, open('Models/LinearModels/lm_ind_lagged.sav', 'wb'))
pickle.dump(lm_mod1, open('Models/LinearModels/lm_ind_all.sav', 'wb'))



#ENET MODELS
#make several different enet models
def make_combos(l1, l2):
    combos = []
    for i in l1:
        for j in l2:
            combos.append((i, j))
    return combos

alphas = [0.1, 0.3, 1, 3]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
combs = make_combos(alphas, l1_ratios)

enet_models_lagged = []
enet_models_all = []
for com in combs:
    mod = Backtest(all_ind_x, ind_y, ElasticNet(com[0], com[1]), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_ind_x2, ind_y, ElasticNet(com[0], com[1]), '2010-01', '2018-01').all_forecasts()
    enet_models_lagged.append(mod)
    enet_models_all.append(mod1)



enet_stats_lagged = model_statistics(enet_models_lagged)
enet_stats_all = model_statistics(enet_models_all)


#DECISION TREES
tree_depth = [5, 10, 20, 30, 50, 100, None]

dtree_models_lagged = []
dtree_models_all = []

for depth in tree_depth:
    mod = Backtest(all_ind_x, ind_y, DecisionTreeRegressor(max_depth=depth), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_ind_x2, ind_y, DecisionTreeRegressor(max_depth=depth), '2010-01', '2018-01').all_forecasts()
    dtree_models_lagged.append(mod)
    dtree_models_all.append(mod1)

pickle.dump(dtree_models_lagged, open('Models/DecisionTreeModels/dtree_ind_lagged.sav', 'wb'))
pickle.dump(dtree_models_all, open('Models/DecisionTreeModels/dtree_ind_all.sav', 'wb'))


dtree_stats_lagged = model_statistics(dtree_models_lagged)
dtree_stats_all = model_statistics(dtree_models_all)


#RANDOM FOREST
num_trees = [10, 15, 20, 30, 50]
rf_models_lagged = []
rf_models_all = []
for num in num_trees:
    mod = Backtest(all_ind_x, ind_y, RandomForestRegressor(n_estimators=num), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_ind_x2, ind_y, RandomForestRegressor(n_estimators=num), '2010-01', '2018-01').all_forecasts()
    rf_models_lagged.append(mod)
    rf_models_all.append(mod1)
    print(num)


pickle.dump(rf_models_lagged, open('Models/RFModels/ind_rf_lagged.sav', 'wb'))
pickle.dump(rf_models_all, open('Models/RFModels/ind_rf_all.sav', 'wb'))


rf_lagged_stats = model_statistics(rf_models_lagged)
rf_all_stats = model_statistics(rf_models_all)



#NEURAL NETWORKS
nn1 = ([30], ['relu'])
nn2 = ([60,30],['relu', 'relu'])
nn3 = ([120,60], ['relu', 'relu'])
nn4 = ([120,120,60], ['relu', 'relu', 'relu'])
nn5 = ([120,60,30], ['relu', 'relu', 'relu'])
nn6 = ([60,30,10], ['relu', 'relu', 'relu'])
nn7 = ([120,60,30,10], ['relu', 'relu', 'relu', 'relu'])
nn8 = ([30,30,30,30], ['relu', 'relu', 'relu', 'relu'])
nn9 = ([120,120,60,60], ['relu', 'relu', 'relu', 'relu'])
nn10 = ([120,60,30,15,5], ['relu', 'relu', 'relu', 'relu', 'relu'])

nns = [nn1, nn2, nn5, nn7, nn10]
nn_models_lagged = []
nn_models_all = []

i=0
for nn in nns:
    mod = Backtest(all_ind_x, ind_y, NeuralNetwork(Sequential, nn[0], nn[1], epoch=1), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_ind_x, ind_y, NeuralNetwork(Sequential, nn[0], nn[1], epoch=1), '2010-01', '2018-01').all_forecasts()
    nn_models_lagged.append(mod)
    nn_models_all.append(mod1)
    i+=1
    print(i)

#this wont save NNs, need to find another way
pickle.dump(nn_models_lagged, open('Models/NNModels/ind_nn_lagged.sav', 'wb'))
pickle.dump(nn_models_all, open('Models/NNModels/ind_nn_all.sav', 'wb'))


nn_stats_lagged = model_statistics(nn_models_lagged)
nn_stats_all = model_statistics(nn_models_all)


#ENSEMBLE

base_learners = [[Naive(), Drift(), PrevailingMean()],
                 [LinearRegression(), ElasticNet(), DecisionTreeRegressor()],
                 [Naive(), Drift(), PrevailingMean(), LinearRegression(), ElasticNet(), DecisionTreeRegressor()]]

ensem_models_lagged = []
ensem_models_all = []

for bases in base_learners:
    mod = EnsembleBacktest(all_ind_x, ind_y, bases, ElasticNet(), '2000-01', '2010-01', '2018-01').all_meta_forecasts()
    mod1 = EnsembleBacktest(all_ind_x2, ind_y, bases, ElasticNet(), '2000-01', '2010-01', '2018-01').all_meta_forecasts()
    ensem_models_lagged.append(mod)
    ensem_models_all.append(mod1)


pickle.dump(ensem_models_lagged, open('Models/EnsembleModels/ind_ensem_lagged.sav', 'wb'))
pickle.dump(ensem_models_all, open('Models/EnsembleModels/ind_ensem_all.sav', 'wb'))


ensem_stats_lagged = model_statistics(ensem_models_lagged)
ensem_stats_all = model_statistics(ensem_models_all)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


#load stock data
stocks = read_ts_csv('Data/1980_top100.csv', format='%Y%m%d')
stocks['ret2'] = stocks['RET'].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0) #gets rid of returns that arent a number
stocks = stocks.pivot_table(index='Date', columns=['TICKER'], values='ret2', aggfunc=sum, fill_value=0)
stocks = stocks*100

#create the lag returns so we can create the predictors
stock_x,  stock_y = lag_returns(stocks)

start, end = '1980-02', '2018-01'

#make all the predictors so we can slice them later
all_stock_x2 = get_all_predictors(stock_x[start:end], mkt[start:end], mkt_st[start:end])
stock_y = stock_y[all_stock_x2[0].index[0]:end]

#make the lagged returns with the same time period
all_stock_x1 = [stock_x[all_stock_x2[0].index[0]:end] for i in range(stock_x.shape[1])]

#saving the data
pickle.dump(all_stock_x1, open('Data/stock_x.sav', 'wb'))
pickle.dump(all_stock_x2, open('Data/stock_x_all_vars.sav', 'wb'))
pickle.dump(stock_y, open('Data/stock_y.sav', 'wb'))



naive_mod = Backtest(all_stock_x1, stock_y, Naive(), '2010-01', '2018-01').all_forecasts()
drift_mod = Backtest(all_stock_x1, stock_y, Drift(), '2010-01', '2018-01').all_forecasts()
pm_mod = Backtest(all_stock_x1, stock_y, PrevailingMean(), '2010-01', '2018-01').all_forecasts()
arima_mod = Backtest(all_stock_x1, stock_y, AutoArima(), '2010-01', '2018-01').all_forecasts()

statmod_stats = model_statistics([naive_mod, drift_mod, pm_mod, arima_mod])

pickle.dump(naive_mod, open('C://Users/mozei/Git/ML_for_EAP/Models/StatModels/naive_stock.sav', 'wb'))
pickle.dump(drift_mod, open('C://Users/mozei/Git/ML_for_EAP/Models/StatModels/drift_stock.sav', 'wb'))
pickle.dump(pm_mod, open('C://Users/mozei/Git/ML_for_EAP/Models/StatModels/pm_stock.sav', 'wb'))
pickle.dump(arima_mod, open('C://Users/mozei/Git/ML_for_EAP/Models/StatModels/arima_stock.sav', 'wb'))

lm_mod = Backtest(all_stock_x1, stock_y, LinearRegression(), '2010-01', '2018-01').all_forecasts()
lm_mod1 = Backtest(all_stock_x2, stock_y, LinearRegression(), '2010-01', '2018-01').all_forecasts()

pickle.dump(lm_mod, open('C://Users/mozei/Git/ML_for_EAP/Models/LinearModels/lm_stock_lagged.sav', 'wb'))
pickle.dump(lm_mod1, open('C://Users/mozei/Git/ML_for_EAP/Models/LinearModels/lm_stock_all.sav', 'wb'))

# ENET MODELS
# make several different enet models
def make_combos(l1, l2):
    combos = []
    for i in l1:
        for j in l2:
            combos.append((i, j))
    return combos


alphas = [0.1, 0.3, 1, 3]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
combs = make_combos(alphas, l1_ratios)

enet_models_lagged = []
enet_models_all = []
for com in combs:
    mod = Backtest(all_stock_x1, stock_y, ElasticNet(com[0], com[1]), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_stock_x2, stock_y, ElasticNet(com[0], com[1]), '2010-01', '2018-01').all_forecasts()
    enet_models_lagged.append(mod)
    enet_models_all.append(mod1)

pickle.dump(enet_models_lagged, open('C://Users/mozei/Git/ML_for_EAP/Models/EnetModels/stocks_enet_lagged.sav', 'wb'))
pickle.dump(enet_models_all, open('C://Users/mozei/Git/ML_for_EAP/Models/EnetModels/stocks_enet_all.sav', 'wb'))

enet_stats_lagged = model_statistics(enet_models_lagged)
enet_stats_all = model_statistics(enet_models_all)

# DECISION TREES
tree_depth = [5, 10, 20, 30, 50, 100, None]

dtree_models_lagged = []
dtree_models_all = []

for depth in tree_depth:
    mod = Backtest(all_stock_x1, stock_y, DecisionTreeRegressor(max_depth=depth), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_stock_x2, stock_y, DecisionTreeRegressor(max_depth=depth), '2010-01', '2018-01').all_forecasts()
    dtree_models_lagged.append(mod)
    dtree_models_all.append(mod1)

pickle.dump(dtree_models_lagged,
            open('C://Users/mozei/Git/ML_for_EAP/Models/DecisionTreeModels/dtree_stocks_lagged.sav', 'wb'))
pickle.dump(dtree_models_all,
            open('C://Users/mozei/Git/ML_for_EAP/Models/DecisionTreeModels/dtree_stocks_all.sav', 'wb'))

dtree_stats_lagged = model_statistics(dtree_models_lagged)
dtree_stats_all = model_statistics(dtree_models_all)

# RANDOM FOREST
num_trees = [10, 15, 20, 30, 50]
rf_models_lagged = []
rf_models_all = []
for num in num_trees:
    mod = Backtest(all_stock_x1, stock_y, RandomForestRegressor(n_estimators=num), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_stock_x2, stock_y, RandomForestRegressor(n_estimators=num), '2010-01',
                    '2018-01').all_forecasts()
    rf_models_lagged.append(mod)
    rf_models_all.append(mod1)
    print(num)

pickle.dump(rf_models_lagged, open('C://Users/mozei/Git/ML_for_EAP/Models/RFModels/rf_stocks_lagged.sav', 'wb'))
pickle.dump(rf_models_all, open('C://Users/mozei/Git/ML_for_EAP/Models/RFModels/rf_stocks_all.sav', 'wb'))

rf_lagged_stats = model_statistics(rf_models_lagged)
rf_all_stats = model_statistics(rf_models_all)

# ENSEMBLE
base_learners = [[Naive(), Drift(), PrevailingMean()],
                 [LinearRegression(), ElasticNet(), DecisionTreeRegressor()],
                 [Naive(), Drift(), PrevailingMean(), LinearRegression(), ElasticNet(), DecisionTreeRegressor()]]

ensem_models_lagged = []
ensem_models_all = []

for bases in base_learners:
    mod = EnsembleBacktest(all_stock_x1, stock_y, bases, ElasticNet(), '2000-01', '2010-01',
                           '2018-01').all_meta_forecasts()
    mod1 = EnsembleBacktest(all_stock_x2, stock_y, bases, ElasticNet(), '2000-01', '2010-01',
                            '2018-01').all_meta_forecasts()
    ensem_models_lagged.append(mod)
    ensem_models_all.append(mod1)

pickle.dump(ensem_models_lagged,
            open('C://Users/mozei/Git/ML_for_EAP/Models/EnsembleModels/ensem_stocks_lagged.sav', 'wb'))
pickle.dump(ensem_models_all, open('C://Users/mozei/Git/ML_for_EAP/Models/EnsembleModels/ensem_stocks_all.sav', 'wb'))

ensem_stats_lagged = model_statistics(ensem_models_lagged)
ensem_stats_all = model_statistics(ensem_models_all)


ns = [nn1, nn2, nn5]


stock_nn_lagged = []
stock_nn_all = []
i=0
for n in ns:
    mod = Backtest(all_stock_x1, stock_y, NeuralNetwork(Sequential, n[0], n[1], epoch=1), '2010-01', '2018-01').all_forecasts()
    mod1 = Backtest(all_stock_x2, stock_y, NeuralNetwork(Sequential, n[0], n[1], epoch=1), '2010-01', '2018-01').all_forecasts()
    stock_nn_lagged.append(mod)
    stock_nn_all.append(mod1)
    i += 1
    print(i)



stock_nn_stats_lagged= model_statistics(stock_nn_lagged)
stock_nn_stats_all = model_statistics(stock_nn_all)



#load in benchmark stuff
index_compare = read_ts_csv('Data/index_compare.csv')


def benchmark_cum_rets(df):
    index = df.index
    df_copy = df.to_numpy()
    cum_rets = 100 * np.cumprod((1 + df / 100), axis=0)
    cum_rets = pd.DataFrame(cum_rets, index = index)

    return cum_rets

rets_test = benchmark_cum_rets(index_compare["2010-01":])