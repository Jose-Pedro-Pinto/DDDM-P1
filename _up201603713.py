def order(past_moves, past_sales, market):
    """ function implementing a simple strategy; parameters:
        * past_sales: list with historical data for the market trend
        * market: function to evaluate the actual sales; 'market(value)' returns:
                - inventory, if smaller than demand (everything was sold)
                - true demand otherwise (i.e., some inventory wasn't sold)
    """
    import pandas as pd
    import numpy as np
    import math
    from statistics import mean
    from tensorflow.keras import backend
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Bidirectional
    from sklearn.preprocessing import MinMaxScaler

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            # shift the data by lookback
            a = dataset[i:(i + look_back), 0]
            # add shifted data to X
            dataX.append(a)
            # add original data do Y
            dataY.append(dataset[i + look_back, 0])
        # return the X and Y data, with the past data and the intended prediction
        return np.array(dataX), np.array(dataY)

    # custom loss function for loss and profit
    def custom_cost_function(y, pred):
        # set cost of unsold item
        cost = 9
        # set profit of sold item
        profit = 1
        # get difference bettwen predicted value and real value
        diff = pred - y
        # return the amount of missed profit(cost for buying too much or too little)
        return backend.mean(
            abs(diff) * ((backend.cast(diff > 0, "float32")) * cost + (backend.cast(diff <= 0, "float32")) * profit))

    # scale the train data as the LSTM is sensitive to scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(np.asarray(past_sales).reshape(-1, 1))
    train = train.astype('float32')

    look_back = 1
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)

    # split into train and warmup sets
    train_size = int(len(trainX) * 0.95)
    trainX, warmX = trainX[0:train_size], trainX[train_size:len(trainX)]
    trainY, warmY = trainY[0:train_size], trainY[train_size:len(trainY)]

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    warmX = np.reshape(warmX, (warmX.shape[0], 1, warmX.shape[1]))

    # number of models to train
    nmodels = 10
    models = []
    # create and fit the LSTM networks
    for i in range(0, nmodels):
        # print progress in training
        print("Model ", i + 1)
        model = Sequential()
        # add input layer(LSTM)
        model.add(Bidirectional(LSTM(4, input_shape=(1, look_back))))
        # add output layer(dense, fully connected)
        model.add(Dense(1))
        # set loss function to custom loss function and optimiser
        model.compile(loss=custom_cost_function, optimizer='adam')
        # train the model
        model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
        # add model to the list of models
        models += [model]

    # start model weights at equal values
    weights = [1] * len(models)
    # use warmup to set the weights for each model
    for i in range(0, len(warmX)):
        # scale the data as the LSTM is sensitive to scale
        sales = scaler.transform(trainY.reshape(-1, 1))
        totalWeight = sum(weights)
        move = 0
        moves = []
        # obtain predictions for all models
        for i in range(0, nmodels):
            # get the model
            model = models[i]
            # get the model weight
            weight = weights[i]

            # get the predicted move
            movei = model.predict(sales[-1].reshape(-1, 1, look_back))
            # scale it back to get the move in the right scale
            movei = scaler.inverse_transform(movei).reshape(1)[0]
            # update final move with this moves weight
            move += movei * (weight / totalWeight)
            # add move to the moves list
            moves += [movei]
        # round move to closest integer
        int(move + .5)
        # add sales to the previous sales
        trainY = np.append(trainY, [min(warmY[i], move)])
        # update the weights
        for i in range(0, nmodels):
            if (move == trainY[-1] and moves[i] > move):
                weights[i] += 1
            if (move > trainY[-1] and moves[i] < move):
                weights[i] += 1
        # increase previous sales value to bring moved predictions back up
        if move == trainY[-1]:
            trainY[-1] *= 1.1

    while True:
        sales = scaler.transform(np.asarray(past_sales).reshape(-1, 1))
        totalWeight = sum(weights)
        move = 0
        moves = []
        for i in range(0, nmodels):
            # get the predicted move
            model = models[i]
            # get the model weight
            weight = weights[i]

            # get the predicted move
            movei = model.predict(sales[-look_back:].reshape(-1, 1, look_back))
            # scale it back to get the move in the right scale
            movei = scaler.inverse_transform(movei).reshape(1)[0]
            # update final move with this moves weight
            move += movei * (weight / totalWeight)
            # add move to the moves list
            moves += [movei]
        # round move to closest integer
        move = int(move + .5)
        # send the move to market and get the number of sold items
        prev_market = market(move)
        if prev_market == None:  # game over
            return
        past_sales = np.append(past_sales, prev_market)
        # update the weights
        for i in range(0, nmodels):
            if (move == past_sales[-1] and moves[i] > move):
                weights[i] += 1
            if (move > past_sales[-1] and moves[i] < move):
                weights[i] += 1
        # increase previous sales value to bring moved predictions back up
        if move == past_sales[-1]:
            past_sales[-1] *= 1.1