#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datetime, os, warnings
from jsonlogger_class import JSONLogger
from sklearn.model_selection import train_test_split, RandomizedSearchCV 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

warnings.filterwarnings('ignore')

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tobet_logs'
PUBLISHPATH = 'publish/'

def newest_predictions(tier) -> str:
    logger.log('info', f'Searching latest {tier} prediction file..')
    files = os.listdir(PUBLISHPATH)

    paths = []
    for basename in files:
       if (basename[0:5] == tier) and ('updated' not in basename):
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

    # Include rolling averages and lag features for 'Goalspergame'
    df['Home_Goals_Rolling_Avg'] = df['Hometeam GpG'].rolling(window=2).mean().shift(1)
    df['Away_Goals_Rolling_Avg'] = df['Awayteam GpG'].rolling(window=2).mean().shift(1)

    # Fill NaN values for rolling averages with original goals per game as fallback
    df['Home_Goals_Rolling_Avg'].fillna(df['Hometeam GpG'], inplace=True)
    df['Away_Goals_Rolling_Avg'].fillna(df['Awayteam GpG'], inplace=True)
    logger.log('info', f'{tier} training/updated files ready..')
    return df

def train_lstmmodel(df):
    logger.log('info', f'Creating DNN model..')
    label_encoder = LabelEncoder()
    df['Prediction_enc'] = label_encoder.fit_transform(df['Prediction'])

    features = ['Hometeam GpG', 'Awayteam GpG', 'History_num', 'History_Missing', 'Prediction_enc', 'Home_Goals_Rolling_Avg', 'Away_Goals_Rolling_Avg']
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

    logger.log('info', f'Training DNN model..')
    # Define a function to create the model, required for KerasClassifier
    def create_model(units=50, dropout_rate=0.2, learning_rate=0.01):
        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Wrap the model with KerasClassifier for use in scikit-learn
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the hyperparameters grid
    param_dist = {
        'units': [42],
        'dropout_rate': [0.5],
        'learning_rate': [0.05],
        'batch_size': [10],
        'epochs': [150]
    }

    # Use RandomizedSearchCV to find the best hyperparameters
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42)
    random_search_result = random_search.fit(X_train, y_train)

    logger.log('info', f'Model Best Parameters: {random_search_result.best_params_}..')
    logger.log('info', f'Model Best Accuracy: {random_search_result.best_score_:.2f}..')

    # Evaluate the model with the best parameters on the test set
    best_model = random_search_result.best_estimator_
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.log('info', f'Model Testing Accuracy: {accuracy:.2f}..')
    logger.log('info', f'Model training completed..')
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
        return perc

    # Apply conversion to the 'History' column
    df['History_num'] = df['History %'].apply(convert_history)
    df['History_Missing'] = df['History %'].isna().astype(int)
    df['History_num'].fillna(0.5, inplace=True)

    # Include rolling averages and lag features for 'Goalspergame'
    df['Home_Goals_Rolling_Avg'] = df['Hometeam GpG'].rolling(window=2).mean().shift(1)
    df['Away_Goals_Rolling_Avg'] = df['Awayteam GpG'].rolling(window=2).mean().shift(1)

    # Fill NaN values for rolling averages with original goals per game as fallback
    df['Home_Goals_Rolling_Avg'].fillna(df['Hometeam GpG'], inplace=True)
    df['Away_Goals_Rolling_Avg'].fillna(df['Awayteam GpG'], inplace=True)

    features = ['Hometeam GpG', 'Awayteam GpG', 'History_num', 'History_Missing', 'Prediction_enc', 'Home_Goals_Rolling_Avg', 'Away_Goals_Rolling_Avg']
    X = df[features]
    X = scaler.fit_transform(X)

    # Reshape data for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    filename = PUBLISHPATH + f'{tier}_lstm_{datesave}.csv'
    df['tobet'] = model.predict(X)
    df[['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'History %', 'Hometeam GpG', 'Awayteam GpG', 'tobet']].to_csv(filename, index=False)
    logger.log('info', f'{tier} file is available..', filename)
    return

def main(tier):
    logger.log('info', f'Process started for {tier}..')
    filename = newest_predictions(tier)
    data = trainset(tier)
    model, encoder, scaler = train_lstmmodel(data)
    predictions(filename,model, encoder, scaler, tier)
    logger.log('info', f'Process completed for {tier}..')
    return


if __name__ == '__main__':
    os.chdir('D:\\Python Apps\\Patreon')
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main('Tier1')
    except Exception as e:
        logger.log('critical', "Exception occured whie running Tier1", info=str(e))

    """ need to update column names
    try:
        main('Tier2') 
    except Exception as e:
        logger.log('critical', "Exception occured whie running Tier2", info=str(e))
    """