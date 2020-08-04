
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLarsIC, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.backend import clear_session
import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs


#############################################################################################
#SKELTON CLASS FOR OUR ML MODELS
#############################################################################################



class Model(object):
    '''
    This is a skeleton class to hold objects. All ML classes will be built of this to handle parameter tuning, fitting,
    and predicting in the same manner for our backtesting and forecasting functions.
    '''

    def __init__(self, model):
        self.model = model()

    def fit(self):
        pass

    def predict(self):
        pass

    def get_coefs(self):
        pass


#############################################################################################
#CLASS FOR OUR BASIC LINEAR REGRESSION MODELS
#############################################################################################

class OlsModel(Model):
    '''
    This is a child class from the parent Model class. This will be used for all Linear Regression models
    using the Sklearn.linear_model package. The 4 models which can be used from this class are LinearRegression(),
    Lasso(), Ridge(), ElasticNet().
    '''

    def __init__(self, model, lamb=1, l1_rat=0.5):
        '''Model is type of Linear Regression model regular/lasso/Ridge/ElasticNet'''
        super().__init__(model)
        self.lamb = lamb
        self.l1_rat = l1_rat


        if self.model == Lasso:
            #tune model with lambda parameter
            self.model = self.model(alpha = self.lamb)
        elif self.model == Ridge:
            self.model = self.model(alpha = self.lamb)
        elif self.model == ElasticNet:
            self.model = self.model(alpha=self.lamb, l1_ratio = self.l1_rat)


    def fit(self, X, y):
        '''Function to fit a lasso regression model using the scipy package'''
        self.model.fit(X, y)
        return self


    def prediction(self, X_test):
        '''Function to make predictons from our fitted model'''
        preds = self.model.predict(X_test)
        return preds



#############################################################################################
# SIMPLE NEURAL NETWORK MODEL
#############################################################################################


class NeuralNetwork(Model):
    '''
    This is a child class of the parent Model class. This class is a simple neural network designed using the
    keras package and tensorflow as the backend.
    '''

    def __init__(self, model, layers_nodes=[], layers_activations=[], reg_penalty = 0.001, epoch=1):
        '''
        :param model: this should be Sequential from keras package
        :param layers_nodes: the number of nodes at each layer. we will add an output layer
        :param layers_activations: the activation function to use at each layer (
        :param reg_penalty: reg penalty on layers
        :param epochs: the number of pass through the data you want the model to go through when fitting
        '''

        super().__init__(model)
        self.layers_nodes = layers_nodes
        self.layers_activations = layers_activations
        self.reg_penalty = reg_penalty
        self.epoch = epoch

    def build_model(self, input_dim, output_dim):
        '''Initialize the graph only once. Running into memory leaks when building multiple grpahs. Also
        K.clear_session() not working in a loop (fails after 3 iters) so this is a workaround
        :param input_dim : the number of features in the model
        :param output_dim : the number of labels given to the model
        '''

        #if no hidden layers then it should be pretty close to linear regression (used as validity check)
        if len(self.layers_nodes) == 0:
            self.model.add(layers.Dense(output_dim, activation='linear', input_shape=[input_dim]))
        else:
            #the first layer which needs an input shape
            self.model.add(layers.Dense(self.layers_nodes[0], activation=self.layers_activations[0], input_shape=[input_dim],
                           kernel_initializer=keras.initializers.glorot_uniform(),
                           kernel_regularizer=keras.regularizers.l1_l2(self.reg_penalty)))
            #the remaining hidden layers
            for i in range(1, len(self.layers_nodes)):
                self.model.add(layers.Dense(self.layers_nodes[i], activation=self.layers_activations[i],
                               kernel_initializer=keras.initializers.glorot_uniform(),
                               kernel_regularizer=keras.regularizers.l1_l2(self.reg_penalty)))
            #the output layer
            self.model.add(layers.Dense(output_dim, activation='linear'))
        #model compile step
        # opt = keras.optimizers.RMSprop(learning_rate=self.learn_rate)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'], loss_weights = [1.0])
        return self


    def fit(self, X, y):
        '''
        Fits the model on the training data.
        :param X: our training features
        :param y: our training labels
        :return: self now fitted on training data
        '''
        #only using RMSprop for now, change chnage or make the optimizer a varible later
        #this is the learning process for backpropogation
        # opt = keras.optimizers.RMSprop(learning_rate=self.learn_rate)

        self.model.fit(X, y, epochs=self.epoch, verbose=0)

        return self


    def predict(self, X_test):
        '''
        Function to get predictions on our test data
        :param X_test: the features of our test data
        :return: our predictions
        '''
        preds = self.model.predict(X_test)
        return preds


###################################################################################################
#OBJECT SO WE CAN RUN NAIVE FORECASTING
###################################################################################################


class Naive(object):
    '''This is an object which does Naive forecasting. Naive forecasting is just the value of t+1 is equal
    to the value at time t. This is a decent forecast to benchmark to since it basically says there is
    nothing predictive about what we know now to predict tomorrow other than that tomorrow=today. This class is made
    solely so it has the same method names as our ML models and can be run in the code.
    '''

    def __init(self, model=None):
        self.model = model

    def fit(self, X, y):
        '''
        For the Naive forecast we will be taking the last row in column y as our and feeding this into the predict
        funciton which will be returning it.
        :param X: Just a placeholder for congruence
        :param y: label data
        :return: self which is the last observation
        '''
        if type(y) == pd.DataFrame:
            self.model = y.iloc[-1].to_numpy()
        else:
            self.model = y[-1]
        return self

    def predict(self, X_test):
        horizon = X_test.shape[0]
        preds_temp = self.model.reshape((1,-1))
        preds = preds_temp
        for i in range(1,horizon):
            preds = np.concatenate([preds, preds_temp], axis=0)
        return preds


class PrevailingMean(object):
    '''This is a class that us the mean of a predetermined period as the forecast. If no period length is specified
    the entire training period will be used as the mean
    '''

    def __init__(self, model=None, period=None):
        self.model = model
        self.period = period #if not None this should be an integer. the int is the number of months you want the mean lookback for

    def fit(self, X, y):
        if self.period != None:
            y = y.tail(self.period)

        self.model = y.mean(axis=0)
        return self

    def predict(self, X_test):
        pred = self.model
        return pred


class Drift(object):

    def __init__(self, period=12):
        """This is a statistical forecasting method based on the average period increase/decrease over time
        return preds and use this as a prediction"""
        self.period = period

    def fit(self, x_train, y_train):
        if type(y_train) == pd.DataFrame or type(y_train) == pd.Series:
            y_train = y_train.to_numpy().reshape((-1,1))
            y_train = y_train[-self.period:]
        else:
            y_train = y_train.reshape((-1,1))
            y_train = y_train[-self.period:]

        self.drift = ((y_train[-1] - y_train[0])/y_train.shape[0]).item()
        self.model = (y_train[-1]).item()
        return self

    def predict(self, x_test):
        if type(x_test) == pd.DataFrame:
            x_test.to_numpy().reshape((-1,1))
        elif type(x_test) != np.ndarray:
            x_test = np.asarray(x_test).reshape((-1,1))
        horizon = np.arange(1, x_test.shape[0]+1)
        preds = self.model + horizon*self.drift
        return preds




class MetaLearner(Model):

    def __init__(self, base_learners, meta_model, validation_size=24):
        '''This is a meta-learner built for the main stacking ensemble.
        :param (base_learners) : a list of statistical or ML models to make a base prediction
        :param (meta_model) : the second level ML model to make wegihted averages of those predictions
        :param (validation_size) : the size of the validation set which is used for the base learners predictions and
                                    the training size of the meta_model
        '''
        self.base_learners = base_learners
        self.meta_model = meta_model
        self.validation_size = validation_size

    def fit(self, x_train, y_train):
        x_valid, x_train = x_train[-self.validation_size:], x_train[:-self.validation_size]
        y_valid, y_train = y_train[-self.validation_size:], y_train[:-self.validation_size]



        #the preds will be for each row, we will transpose this later
        meta_featutres = np.zeros((len(self.base_learners), self.validation_size))
        for i in range(len(self.base_learners)):
            self.base_learners[i].fit(x_train, y_train)
            preds = self.base_learners[i].predict(x_valid)
            meta_featutres[i] = preds.ravel()

        #this will be the features in the meta learner
        meta_featutres = np.transpose(meta_featutres)
        self.meta_model.fit(meta_featutres, y_valid)
        return self

    def predict(self, x_test):
        meta_features = np.zeros((len(self.base_learners), x_test.shape[0]))
        for i in range(len(self.base_learners)):
            base_preds = self.base_learners[i].predict(x_test)
            meta_features[i] = base_preds
        meta_features = np.transpose(meta_features)
        preds = self.meta_model.predict(meta_features)
        return preds


class LogReg(Model):

    def __init__(self, probs=True):
        '''This is a class to run Logisitc Regression using the sklearn package.
        In this model we can either run predictions using the probabilities or run predicitons
        using the classification
        :param probs : if True returns probablilities, if False returns class predicitions'''

        self.model = LogisticRegression()
        self.probs = probs

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        if self.probs == True:
            #this will return an array of shape [1,2] in the order of [prob for 0, prob for 1]; so we
            # ravel to make if 0d-array and take the 1 prob column
            preds = self.model.predict_proba(x_test).ravel()
            preds = preds[1]
        else:
            preds = self.model.predict(x_test)

        return preds


class AutoArima():
    def __init__(self):
        '''Class built on the pmdarima package. This class is designed to run like the auto arima
        funciton in R. Using the auto arima function we can create an ARIMA model to make forecasting
        predicitons on our data.
        '''
        self.model = None

    def find_order(self, data):
        # determine number of differences
        kpss_diffs = ndiffs(data, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(kpss_diffs, adf_diffs)
        # grid search to find order
        self.order = pm.auto_arima(data, d=n_diffs, seasonal=False, stepwise=True,
                              suppress_warnings=True, error_action="ignore",
                              max_order=None, trace=True, maxiter=20).order
        self.model = pm.arima.ARIMA(order=self.order)
        return self

    def build_model(self, y_train):
        order = self.find_order(y_train)
        self.model = pm.ARIMA(order)
        return self


    def fit(self, x_train, y_train):
        self.model.fit(y_train)
        return self

    def predict(self, x_test):
        # 1 means one step forecast
        preds = self.model.predict(1)
        return preds







