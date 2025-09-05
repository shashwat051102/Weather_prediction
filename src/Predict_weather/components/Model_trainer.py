import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.Predict_weather import logger
from src.Predict_weather.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def create_sequences(self, data, seq_length):
        
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
            
        return np.array(xs), np.array(ys)
    
    
    def train(self):
        
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        logger.info(f"Train data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")
        target_column = self.config.target_column
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data[[target_column]])
        test_scaled = scaler.transform(test_data[[target_column]])
        
        X_train, y_train = self.create_sequences(train_scaled, self.config.sequence_length)
        X_test, y_test = self.create_sequences(test_scaled, self.config.sequence_length)
        
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(self.config.hidden_size, return_sequences=True, input_shape=(self.config.sequence_length, 1)))
        model.add(Dropout(self.config.dropout))
        model.add(LSTM(self.config.hidden_size))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(1))
        
        model.compile(optimizer = Adam(learning_rate=self.config.learning_rate), loss='mse', metrics=['mae'])
        
        logger.info("Starting model training...")
        
        history = model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_test, y_test)
        )
        
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        scaler_path = os.path.join(self.config.root_dir, "scaler.pkl")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Scaler saved at: {scaler_path}")
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Final Training Loss: {final_loss}, Final Validation Loss: {final_val_loss}")
        
        print(f"Model training completed!")
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")