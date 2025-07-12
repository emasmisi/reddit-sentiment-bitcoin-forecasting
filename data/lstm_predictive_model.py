#!/usr/bin/env python3
"""
🚀 LSTM PREDICTIVE MODEL - FINAL IMPLEMENTATION
===============================================
PHASE 2: Modello predittivo finale per la tesi
- LSTM + FinBERT sentiment per predizione prezzi
- Focus su asset con correlazioni più forti (Bitcoin, S&P 500)
- Implementation rapida per tesi triennale
- Performance metrics e validation completa
"""

import pandas as pd
import numpy as np
import sqlite3
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ML utilities
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialLSTMPredictor:
    """🚀 Modello LSTM per predizione prezzi basato su sentiment"""
    
    def __init__(self):
        print("🚀 === FINANCIAL LSTM PREDICTOR ===")
        print("🎯 PHASE 2: Modello predittivo finale\n")
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.datasets = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_and_prepare_data(self, asset_name, subreddit_filter=None):
        """📊 Carica e prepara dati per training LSTM"""
        
        print(f"📊 Loading data for {asset_name.upper()}...")
        
        # Load sentiment data
        sentiment_files = glob.glob("data/processed/finbert_sentiment_analysis_*.csv")
        if not sentiment_files:
            print("❌ No sentiment data found!")
            return None
            
        sentiment_df = pd.read_csv(sorted(sentiment_files)[-1])
        sentiment_df['created_date'] = pd.to_datetime(sentiment_df['created_date'])
        
        # Filter by subreddit if specified
        if subreddit_filter:
            sentiment_df = sentiment_df[sentiment_df['subreddit'] == subreddit_filter]
            print(f"   📊 Using sentiment from r/{subreddit_filter}")
        
        # Daily sentiment aggregation
        daily_sentiment = sentiment_df.groupby('created_date').agg({
            'final_score': 'mean',
            'confidence': 'mean',
            'finbert_score': 'mean',
            'vader_compound': 'mean',
            'bullish_terms': 'sum',
            'bearish_terms': 'sum',
            'has_ticker': 'sum',
            'id': 'count'
        }).reset_index()
        
        daily_sentiment.columns = [
            'Date', 'sentiment_mean', 'confidence', 'finbert_score', 
            'vader_compound', 'bullish_terms', 'bearish_terms', 
            'ticker_mentions', 'post_volume'
        ]
        
        # Add derived sentiment features
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'].diff()
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_mean'].rolling(3).std()
        daily_sentiment['bullish_ratio'] = (
            daily_sentiment['bullish_terms'] / 
            (daily_sentiment['bullish_terms'] + daily_sentiment['bearish_terms'] + 1)
        )
        
        # Load financial data
        financial_files = glob.glob(f"data/financial/{asset_name}_*.csv")
        if not financial_files:
            print(f"❌ No financial data found for {asset_name}")
            return None
            
        financial_df = pd.read_csv(sorted(financial_files)[-1])
        financial_df['Datetime'] = pd.to_datetime(financial_df['Datetime'], utc=True)
        financial_df['Date'] = financial_df['Datetime'].dt.date
        financial_df['Date'] = pd.to_datetime(financial_df['Date'])
        
        # Daily financial aggregation
        daily_financial = financial_df.groupby('Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
        
        # Add financial features
        daily_financial['daily_return'] = daily_financial['Close'].pct_change()
        daily_financial['volatility'] = daily_financial['daily_return'].rolling(5).std()
        daily_financial['price_ma_5'] = daily_financial['Close'].rolling(5).mean()
        daily_financial['price_ma_10'] = daily_financial['Close'].rolling(10).mean()
        daily_financial['volume_ma'] = daily_financial['Volume'].rolling(5).mean()
        
        # Technical indicators
        daily_financial['rsi'] = self._calculate_rsi(daily_financial['Close'])
        daily_financial['bb_upper'], daily_financial['bb_lower'] = self._calculate_bollinger_bands(daily_financial['Close'])
        
        # Merge datasets
        merged_df = daily_sentiment.merge(daily_financial, on='Date', how='inner')
        
        if len(merged_df) == 0:
            print(f"❌ No overlapping data for {asset_name}")
            return None
            
        print(f"   📅 Data period: {merged_df['Date'].min().date()} → {merged_df['Date'].max().date()}")
        print(f"   📊 Total samples: {len(merged_df)}")
        
        # Remove NaN values
        merged_df = merged_df.dropna()
        print(f"   🧹 After cleaning: {len(merged_df)} samples")
        
        return merged_df
        
    def _calculate_rsi(self, prices, window=14):
        """📈 Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """📊 Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
        
    def prepare_lstm_sequences(self, data, target_col='Close', sequence_length=10):
        """🔄 Prepara sequenze per LSTM"""
        
        print(f"🔄 Preparing LSTM sequences (length={sequence_length})...")
        
        # Select features based on correlation results
        feature_columns = [
            # Sentiment features (from correlation analysis)
            'sentiment_mean', 'confidence', 'finbert_score', 'sentiment_momentum',
            'bullish_terms', 'bearish_terms', 'bullish_ratio', 'post_volume',
            
            # Financial features
            'Open', 'High', 'Low', 'Volume', 'daily_return', 'volatility',
            'price_ma_5', 'price_ma_10', 'rsi'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"   📊 Using features: {len(available_features)}")
        
        # Prepare feature matrix
        feature_data = data[available_features].values
        target_data = data[target_col].values
        
        # Scale features
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        feature_scaled = feature_scaler.fit_transform(feature_data)
        target_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(feature_scaled)):
            X.append(feature_scaled[i-sequence_length:i])
            y.append(target_scaled[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"   📦 Sequences created: X{X.shape}, y{y.shape}")
        
        return X, y, feature_scaler, target_scaler, available_features
        
    def build_lstm_model(self, input_shape, model_type='advanced'):
        """🏗️ Costruisce architettura LSTM"""
        
        print(f"🏗️ Building LSTM model ({model_type})...")
        
        model = Sequential()
        
        if model_type == 'simple':
            # Simple LSTM for quick training
            model.add(LSTM(50, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            
        elif model_type == 'advanced':
            # Advanced LSTM with multiple layers
            model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            
            model.add(LSTM(25))
            model.add(Dropout(0.2))
            
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   🔧 Model compiled with {model.count_params():,} parameters")
        
        return model
        
    def train_model(self, asset_name, subreddit_filter=None, sequence_length=10, model_type='advanced'):
        """🚀 Training completo del modello"""
        
        print(f"\n🚀 === TRAINING {asset_name.upper()} MODEL ===")
        
        # Load and prepare data
        data = self.load_and_prepare_data(asset_name, subreddit_filter)
        if data is None:
            return None
            
        # Prepare LSTM sequences
        X, y, feature_scaler, target_scaler, features = self.prepare_lstm_sequences(
            data, sequence_length=sequence_length
        )
        
        if len(X) < 20:
            print(f"❌ Insufficient data for training ({len(X)} samples)")
            return None
            
        # Train-test split (temporal split)
        split_index = int(0.8 * len(X))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"   📊 Train samples: {len(X_train)}")
        print(f"   📊 Test samples: {len(X_test)}")
        
        # Build model
        model = self.build_lstm_model(
            input_shape=(sequence_length, X.shape[2]),
            model_type=model_type
        )
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        print(f"   🏃 Starting training...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = target_scaler.inverse_transform(train_pred)
        test_pred = target_scaler.inverse_transform(test_pred)
        y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train_actual, train_pred)
        test_metrics = self._calculate_metrics(y_test_actual, test_pred)
        
        print(f"\n📊 === TRAINING RESULTS ===")
        print(f"🏋️ Train Metrics:")
        print(f"   RMSE: ${train_metrics['rmse']:.2f}")
        print(f"   MAE:  ${train_metrics['mae']:.2f}")
        print(f"   R²:   {train_metrics['r2']:.4f}")
        
        print(f"🎯 Test Metrics:")
        print(f"   RMSE: ${test_metrics['rmse']:.2f}")
        print(f"   MAE:  ${test_metrics['mae']:.2f}")
        print(f"   R²:   {test_metrics['r2']:.4f}")
        
        # Direction accuracy
        train_direction_acc = self._calculate_direction_accuracy(y_train_actual, train_pred)
        test_direction_acc = self._calculate_direction_accuracy(y_test_actual, test_pred)
        
        print(f"🎯 Direction Accuracy:")
        print(f"   Train: {train_direction_acc:.2%}")
        print(f"   Test:  {test_direction_acc:.2%}")
        
        # Store results
        model_key = f"{asset_name}_{subreddit_filter if subreddit_filter else 'overall'}"
        
        self.models[model_key] = model
        self.scalers[model_key] = {'feature': feature_scaler, 'target': target_scaler}
        self.datasets[model_key] = {
            'data': data,
            'features': features,
            'X_train': X_train, 'X_test': X_test,
            'y_train_actual': y_train_actual, 'y_test_actual': y_test_actual,
            'train_pred': train_pred, 'test_pred': test_pred
        }
        
        self.results[model_key] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_direction_acc': train_direction_acc,
            'test_direction_acc': test_direction_acc,
            'history': history,
            'sequence_length': sequence_length
        }
        
        return self.results[model_key]
        
    def _calculate_metrics(self, actual, predicted):
        """📊 Calcola metriche di performance"""
        actual = actual.flatten()
        predicted = predicted.flatten()
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
        
    def _calculate_direction_accuracy(self, actual, predicted):
        """🎯 Calcola accuratezza direzione movimento"""
        actual_diff = np.diff(actual.flatten())
        pred_diff = np.diff(predicted.flatten())
        
        actual_direction = np.sign(actual_diff)
        pred_direction = np.sign(pred_diff)
        
        correct = np.sum(actual_direction == pred_direction)
        total = len(actual_direction)
        
        return correct / total if total > 0 else 0
        
    def create_predictions_visualization(self, model_key):
        """📈 Crea visualizzazione predizioni"""
        
        if model_key not in self.results:
            print(f"❌ No results found for {model_key}")
            return
            
        dataset = self.datasets[model_key]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'LSTM Predictions: {model_key.upper()}', fontsize=16, fontweight='bold')
        
        # Training predictions
        axes[0,0].plot(dataset['y_train_actual'], label='Actual', alpha=0.7)
        axes[0,0].plot(dataset['train_pred'], label='Predicted', alpha=0.7)
        axes[0,0].set_title('Training Set Predictions')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Test predictions
        axes[0,1].plot(dataset['y_test_actual'], label='Actual', alpha=0.7)
        axes[0,1].plot(dataset['test_pred'], label='Predicted', alpha=0.7)
        axes[0,1].set_title('Test Set Predictions')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Training loss
        history = self.results[model_key]['history']
        axes[1,0].plot(history.history['loss'], label='Training Loss')
        axes[1,0].plot(history.history['val_loss'], label='Validation Loss')
        axes[1,0].set_title('Model Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1,1].scatter(dataset['y_test_actual'], dataset['test_pred'], alpha=0.6)
        axes[1,1].plot([dataset['y_test_actual'].min(), dataset['y_test_actual'].max()],
                      [dataset['y_test_actual'].min(), dataset['y_test_actual'].max()], 'r--')
        axes[1,1].set_xlabel('Actual Prices')
        axes[1,1].set_ylabel('Predicted Prices')
        axes[1,1].set_title('Actual vs Predicted (Test Set)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('results/lstm_predictions', exist_ok=True)
        plt.savefig(f'results/lstm_predictions/{model_key}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualization saved: results/lstm_predictions/{model_key}_predictions.png")
        
    def generate_trading_signals(self, model_key, confidence_threshold=0.02):
        """📈 Genera segnali di trading"""
        
        if model_key not in self.results:
            print(f"❌ No results found for {model_key}")
            return None
            
        dataset = self.datasets[model_key]
        
        # Calculate price changes
        actual_changes = np.diff(dataset['y_test_actual'].flatten())
        pred_changes = np.diff(dataset['test_pred'].flatten())
        
        # Generate signals
        signals = []
        for i, pred_change in enumerate(pred_changes):
            if abs(pred_change) > confidence_threshold:
                if pred_change > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
            else:
                signal = 'HOLD'
                
            signals.append({
                'day': i,
                'signal': signal,
                'predicted_change': pred_change,
                'actual_change': actual_changes[i] if i < len(actual_changes) else 0,
                'correct': (pred_change > 0) == (actual_changes[i] > 0) if i < len(actual_changes) else False
            })
            
        signals_df = pd.DataFrame(signals)
        
        # Calculate signal accuracy
        signal_accuracy = signals_df['correct'].mean()
        
        # Calculate trading performance
        buy_signals = signals_df[signals_df['signal'] == 'BUY']
        sell_signals = signals_df[signals_df['signal'] == 'SELL']
        
        buy_success_rate = buy_signals['correct'].mean() if len(buy_signals) > 0 else 0
        sell_success_rate = sell_signals['correct'].mean() if len(sell_signals) > 0 else 0
        
        trading_results = {
            'total_signals': len(signals_df),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'overall_accuracy': signal_accuracy,
            'buy_success_rate': buy_success_rate,
            'sell_success_rate': sell_success_rate,
            'signals': signals_df
        }
        
        print(f"\n📈 === TRADING SIGNALS ANALYSIS ===")
        print(f"🎯 Total signals: {trading_results['total_signals']}")
        print(f"📈 Buy signals: {trading_results['buy_signals']}")
        print(f"📉 Sell signals: {trading_results['sell_signals']}")
        print(f"🎯 Overall accuracy: {trading_results['overall_accuracy']:.2%}")
        print(f"📈 Buy success rate: {trading_results['buy_success_rate']:.2%}")
        print(f"📉 Sell success rate: {trading_results['sell_success_rate']:.2%}")
        
        return trading_results
        
    def run_comprehensive_modeling(self):
        """🚀 Esegue modeling completo per i migliori asset"""
        
        print("🚀 === COMPREHENSIVE LSTM MODELING ===")
        print("🎯 Focus su asset con correlazioni più forti\n")
        
        # Priority models based on correlation results
        priority_models = [
            # Crypto (strong correlations, perfect timing)
            ('bitcoin', 'CryptoCurrency'),
            ('ethereum', 'CryptoCurrency'),
            
            # Stocks (strongest correlations)
            ('sp500', 'wallstreetbets'),
            ('nasdaq', 'wallstreetbets')
        ]
        
        results_summary = []
        
        for asset, subreddit in priority_models:
            print(f"\n{'='*60}")
            
            try:
                result = self.train_model(
                    asset_name=asset,
                    subreddit_filter=subreddit,
                    sequence_length=5,  # Shorter for limited data
                    model_type='advanced'
                )
                
                if result:
                    model_key = f"{asset}_{subreddit}"
                    
                    # Create visualizations
                    self.create_predictions_visualization(model_key)
                    
                    # Generate trading signals
                    trading_results = self.generate_trading_signals(model_key)
                    
                    results_summary.append({
                        'asset': asset,
                        'subreddit': subreddit,
                        'test_rmse': result['test_metrics']['rmse'],
                        'test_r2': result['test_metrics']['r2'],
                        'direction_accuracy': result['test_direction_acc'],
                        'trading_accuracy': trading_results['overall_accuracy'] if trading_results else 0
                    })
                    
                else:
                    print(f"❌ Failed to train model for {asset}")
                    
            except Exception as e:
                print(f"❌ Error training {asset}: {e}")
                
        # Final summary
        self._print_final_summary(results_summary)
        
        return results_summary
        
    def _print_final_summary(self, results_summary):
        """📋 Stampa summary finale"""
        
        print(f"\n📋 === FINAL LSTM MODELING SUMMARY ===")
        print("="*60)
        
        if not results_summary:
            print("❌ No successful models trained")
            return
            
        print(f"✅ Successfully trained {len(results_summary)} models\n")
        
        for i, result in enumerate(results_summary, 1):
            print(f"{i}. {result['asset'].upper()} (r/{result['subreddit']}):")
            print(f"   📊 Test RMSE: ${result['test_rmse']:.2f}")
            print(f"   📊 R² Score: {result['test_r2']:.4f}")
            print(f"   🎯 Direction Accuracy: {result['direction_accuracy']:.2%}")
            print(f"   📈 Trading Accuracy: {result['trading_accuracy']:.2%}")
            print()
            
        # Best performing model
        best_model = max(results_summary, key=lambda x: x['direction_accuracy'])
        
        print(f"🏆 BEST PERFORMING MODEL:")
        print(f"   Asset: {best_model['asset'].upper()}")
        print(f"   Subreddit: r/{best_model['subreddit']}")
        print(f"   Direction Accuracy: {best_model['direction_accuracy']:.2%}")
        print(f"   Trading Accuracy: {best_model['trading_accuracy']:.2%}")
        
        print(f"\n🎉 === LSTM MODELING COMPLETED ===")
        print(f"📊 Models ready for tesi demonstration!")
        print(f"📈 Visualizations saved in results/lstm_predictions/")

def main():
    """🚀 Main execution"""
    
    print("🚀 === FINANCIAL LSTM PREDICTOR ===")
    print("🎯 Final predictive model for thesis\n")
    
    try:
        predictor = FinancialLSTMPredictor()
        results = predictor.run_comprehensive_modeling()
        
        if results:
            print(f"\n✅ === SUCCESS ===")
            print(f"🎯 LSTM models trained and validated")
            print(f"📊 Performance metrics calculated")
            print(f"📈 Trading signals generated")
            print(f"🎨 Visualizations created")
            print(f"\n🎓 READY FOR THESIS PRESENTATION! 🎓")
            
        else:
            print(f"\n❌ No models successfully trained")
            
    except Exception as e:
        print(f"❌ Error in LSTM modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()