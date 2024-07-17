#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datetime, os, warnings, time
from jsonlogger_class import JSONLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from skopt import BayesSearchCV
warnings.filterwarnings('ignore')

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tobet_logs'
PUBLISHPATH = 'publish/'

def newest_predictions(tier) -> str:
    logger.log('info', f'Searching latest {tier} prediction file..')
    files = os.listdir(PUBLISHPATH)

    paths = []
    for basename in files:
       if (basename[0:5] == tier) and ('updated' not in basename) and ('lstm' not in basename):
            paths.append(os.path.join(PUBLISHPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        logger.log('info', f'File found..', file)
        return file.split('_')[-1]
    except:
        logger.log('error', f'File not found..')
        return('\\99999999')

def trainset(tier) -> pd:
    logger.log('info', f'Searching {tier} training/updated files..')
    files = os.listdir(PUBLISHPATH)

    df = pd.DataFrame()
    for basename in files:
        if f'{tier}_updated' in basename:
            temp = pd.read_csv(os.path.join(PUBLISHPATH, basename))
            df = pd.concat([df, temp], ignore_index=True)
    df = df.loc[~df['Prediction_Accuracy'].isnull()]
    
    def convert_history(history_str):
            try:
                correct, total = map(int, history_str[1:].split('/'))
                perc = correct / total
            except:
                perc = history_str
            return perc

    # Apply conversion to the 'History' column
    df['History_num'] = df['History %'].apply(convert_history)
    df['History_Missing'] = df['History %'].isna().astype(int)
    df['History_num'].fillna(0.5, inplace=True)

    df['Prediction_Accuracy'] = df['Prediction_Accuracy'].astype(bool)

    if tier == 'Tier1':
        # Include rolling averages and lag features for 'Goalspergame'
        df['Home_Goals_Rolling_Avg'] = df['Hometeam GpG'].rolling(window=3).mean().shift(1)
        df['Away_Goals_Rolling_Avg'] = df['Awayteam GpG'].rolling(window=3).mean().shift(1)

        # Fill NaN values for rolling averages with original goals per game as fallback
        df['Home_Goals_Rolling_Avg'].fillna(df['Hometeam GpG'], inplace=True)
        df['Away_Goals_Rolling_Avg'].fillna(df['Awayteam GpG'], inplace=True)

    else:
        df['Rolling_Home_WpG'] = df['Hometeam WpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Home_DpG'] = df['Hometeam DpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Home_LpG'] = df['Hometeam LpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_WpG'] = df['Awayteam WpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_DpG'] = df['Awayteam DpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_LpG'] = df['Awayteam LpG'].rolling(window=3).mean().shift(1)

    logger.log('info', f'{tier} training/updated files ready..')
    return df

def train_lstmmodel(df, tier):
    logger.log('info', f'Creating DNN model for {tier}..')
    timestart = time.time()
    label_encoder = LabelEncoder()
    df['Prediction_enc'] = label_encoder.fit_transform(df['Prediction'])

    if tier == 'Tier1':
        features = ['Hometeam GpG', 'Awayteam GpG', 
                    'History_num', 'History_Missing', 'Prediction_enc', 
                    'Home_Goals_Rolling_Avg', 'Away_Goals_Rolling_Avg']
    else:
        features = ['Hometeam WpG', 'Hometeam DpG', 'Hometeam LpG', 
                    'Awayteam WpG', 'Awayteam DpG', 'Awayteam LpG', 
                    'History_num', 'History_Missing', 'Prediction_enc', 
                    'Rolling_Home_WpG', 'Rolling_Home_DpG', 'Rolling_Home_LpG',
                    'Rolling_Away_WpG', 'Rolling_Away_DpG', 'Rolling_Away_LpG']
        
    target = 'Prediction_Accuracy'

    X = df[features]
    y = df[target]

    # Normalize the features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape data for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.log('info', f'Training DNN model for {tier}..')

    # Define your custom loss function here
    def custom_loss(y_true, y_pred):
        penalty_factor = 10.0  # Adjust this based on how strict you want to be
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        # Apply penalty to the loss
        loss *= penalty_factor
        return loss

    # Define a function to create the model, required for KerasClassifier
    def create_model(units=50, dropout_rate=0.2, learning_rate=0.01):
        model = Sequential()
        model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Wrap the model with KerasClassifier for use in scikit-learn
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the initial broad hyperparameters grid
    param_dist = {
    'units': (30, 100),             # Range for number of units
    'dropout_rate': (0.1, 0.5),     # Range for dropout rate
    'learning_rate': (1e-4, 1e-2, 'log-uniform'),  # Range for learning rate (log scale)
    'batch_size': (10, 30),         # Discrete choices for batch size
    'epochs': (100, 200)            # Range for number of epochs
    }

    # Search for the best hyperparameters
    bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_dist,
    scoring='accuracy',  # You can use other metrics here such as 'precision', 'recall', etc.
    cv=3,
    n_iter=20,  # Number of parameter settings that are sampled
    n_jobs=-1,  # Use all available CPUs
    verbose=1
    )
    
    bayes_search_result = bayes_search.fit(X_train, y_train)

    logger.log('info', f'{tier} Model Best Parameters: {bayes_search_result.best_params_}..')
    logger.log('info', f'{tier} Model Best Accuracy: {bayes_search_result.best_score_:.2f}..')

    # Evaluate the model with the best parameters on the test set
    best_model = bayes_search_result.best_estimator_
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    logger.log('info', f'{tier} Model Testing Accuracy: {accuracy:.2f}..')
    logger.log('info', f'{tier} Model training completed in {((time.time()-timestart)/60):.1f} mins..')
    return best_model, label_encoder, scaler

def predictions(filedateformat, model, encoder, scaler, tier):
    logger.log('info', f'Creating DNN based predictions for {tier}..')
    df = pd.read_csv(PUBLISHPATH + f'{tier}_{filedateformat}')

    df['Prediction_enc'] = encoder.fit_transform(df['Prediction'])

    def convert_history(history_str):
        try:
            correct, total = map(int, history_str[1:].split('/'))
            perc = correct / total
        except:
            perc = history_str
            if perc == '-':
                perc = None
        return perc

    # Apply conversion to the 'History' column
    df['History_num'] = df['History %'].apply(convert_history)
    df['History_Missing'] = df['History_num'].isna().astype(int)
    df['History_num'].fillna(0.5, inplace=True)

    if tier == 'Tier1':
        # Include rolling averages and lag features for 'Goalspergame'
        df['Home_Goals_Rolling_Avg'] = df['Hometeam GpG'].rolling(window=2).mean().shift(1)
        df['Away_Goals_Rolling_Avg'] = df['Awayteam GpG'].rolling(window=2).mean().shift(1)

        # Fill NaN values for rolling averages with original goals per game as fallback
        df['Home_Goals_Rolling_Avg'].fillna(df['Hometeam GpG'], inplace=True)
        df['Away_Goals_Rolling_Avg'].fillna(df['Awayteam GpG'], inplace=True)

        features = ['Hometeam GpG', 'Awayteam GpG', 
                    'History_num', 'History_Missing', 'Prediction_enc', 
                    'Home_Goals_Rolling_Avg', 'Away_Goals_Rolling_Avg']

    else:
        df['Rolling_Home_WpG'] = df['Hometeam WpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Home_DpG'] = df['Hometeam DpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Home_LpG'] = df['Hometeam LpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_WpG'] = df['Awayteam WpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_DpG'] = df['Awayteam DpG'].rolling(window=3).mean().shift(1)
        df['Rolling_Away_LpG'] = df['Awayteam LpG'].rolling(window=3).mean().shift(1)    

        features = ['Hometeam WpG', 'Hometeam DpG', 'Hometeam LpG', 
                    'Awayteam WpG', 'Awayteam DpG', 'Awayteam LpG', 
                    'History_num', 'History_Missing', 'Prediction_enc', 
                    'Rolling_Home_WpG', 'Rolling_Home_DpG', 'Rolling_Home_LpG',
                    'Rolling_Away_WpG', 'Rolling_Away_DpG', 'Rolling_Away_LpG']

    X = df[features]
    X = scaler.fit_transform(X)

    # Reshape data for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    filename = PUBLISHPATH + f'{tier}_lstm_{datesave}.csv'
    df['tobet'] = model.predict(X)

    if tier == 'Tier1':
        df[['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'History %', 'Hometeam GpG', 'Awayteam GpG', 'tobet']].to_csv(filename, index=False)
    else:
        df[['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'History %', 'Hometeam WpG', 'Hometeam DpG', 'Hometeam LpG', 'Awayteam WpG', 'Awayteam DpG', 'Awayteam LpG', 'tobet']].to_csv(filename, index=False)

    logger.log('info', f'{tier} file is available..', filename)
    return

def main(tier):
    logger.log('info', f'Process started for {tier}..')
    filename = newest_predictions(tier)
    data = trainset(tier)
    model, encoder, scaler = train_lstmmodel(data, tier)
    predictions(filename, model, encoder, scaler, tier)
    logger.log('info', f'Process completed for {tier}..')
    return


if __name__ == '__main__':
    #os.chdir('D:\\Python Apps\\Patreon')
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main('Tier1')
    except Exception as e:
        logger.log('critical', "Exception occured whie running Tier1", info=str(e))

    try:
        main('Tier2') 
    except Exception as e:
        logger.log('critical', "Exception occured whie running Tier2", info=str(e))