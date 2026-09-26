#!/usr/bin/env python3
"""
🚀 LSTM MODEL WITH INTEGRATED DATA FIX
======================================
SOLUZIONE COMPLETA: Fix dati finanziari + LSTM training
- Auto-fix per problemi string/numeric
- LSTM implementation completa
- Ready per tesi triennale
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
import os

class FinancialLSTMWithFix:
    """🚀 LSTM Predictor con auto-fix dei dati"""
    
    def __init__(self):
        print("🚀 === FINANCIAL LSTM WITH INTEGRATED FIX ===")
        print("🔧 Auto-fixing data issues + LSTM training\n")
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.datasets = {}
        
        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def fix_financial_data(self, df):
        """🔧 Fix automatico dati finanziari"""
        
        print("   🔧 Auto-fixing financial data...")
        
        # Convert datetime
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            
        # Fix numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        for col in numeric_columns:
            if col in df.columns:
                # Force numeric conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Remove rows with missing essential data
        essential_cols = ['Open', 'High', 'Low', 'Close']
        available_essential = [col for col in essential_cols if col in df.columns]
        
        if available_essential:
            before_len = len(df)
            df = df.dropna(subset=available_essential, how='all')
            after_len = len(df)
            
            if before_len != after_len:
                print(f"   🧹 Removed {before_len - after_len} invalid rows")
                
        return df
        
    def load_and_prepare_data(self, asset_name, subreddit_filter=None):
        """📊 Carica e prepara dati con auto-fix"""
        
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
        
        # Add derived sentiment features (safe operations)
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'].diff().fillna(0)
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_mean'].rolling(3).std().fillna(0)
        
        # Safe bullish ratio calculation
        total_terms = daily_sentiment['bullish_terms'] + daily_sentiment['bearish_terms']
        daily_sentiment['bullish_ratio'] = np.where(
            total_terms > 0,
            daily_sentiment['bullish_terms'] / total_terms,
            0.5
        )
        
        # Load financial data
        financial_files = glob.glob(f"data/financial/{asset_name}_*.csv")
        if not financial_files:
            print(f"❌ No financial data found for {asset_name}")
            return None
            
        financial_df = pd.read_csv(sorted(financial_files)[-1])
        
        # AUTO-FIX FINANCIAL DATA
        financial_df = self.fix_financial_data(financial_df)
        
        # Process dates safely
        financial_df['Datetime'] = pd.to_datetime(financial_df['Datetime'], utc=True, errors='coerce')
        financial_df['Date'] = financial_df['Datetime'].dt.date
        financial_df['Date'] = pd.to_datetime(financial_df['Date'])
        
        # Daily financial aggregation with error handling
        try:
            daily_financial = financial_df.groupby('Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
            
            # Safe financial calculations
            daily_financial['daily_return'] = daily_financial['Close'].pct_change().fillna(0)
            daily_financial['volatility'] = daily_financial['daily_return'].rolling(5).std().fillna(0)
            daily_financial['price_ma_5'] = daily_financial['Close'].rolling(5).mean()
            daily_financial['volume_ma'] = daily_financial['Volume'].rolling(5).mean().fillna(daily_financial['Volume'])
            
            # Simple technical indicators
            daily_financial['price_change'] = daily_financial['Close'].diff().fillna(0)
            daily_financial['volume_change'] = daily_financial['Volume'].pct_change().fillna(0)
            
        except Exception as e:
            print(f"   ⚠️ Warning in financial calculations: {e}")
            return None
            
        # Merge datasets
        merged_df = daily_sentiment.merge(daily_financial, on='Date', how='inner')
        
        if len(merged_df) == 0:
            print(f"❌ No overlapping data for {asset_name}")
            return None
            
        print(f"   📅 Data period: {merged_df['Date'].min().date()} → {merged_df['Date'].max().date()}")
        print(f"   📊 Total samples: {len(merged_df)}")
        
        # Final cleanup
        merged_df = merged_df.dropna()
        print(f"   🧹 After cleaning: {len(merged_df)} samples")
        
        if len(merged_df) < 10:
            print(f"❌ Insufficient data after cleaning ({len(merged_df)} samples)")
            return None
            
        return merged_df
        
    def prepare_simple_lstm_data(self, data, target_col='Close', lookback=3):
        """🔄 Prepara dati LSTM semplificati"""
        
        print(f"🔄 Preparing LSTM data (lookback={lookback})...")
        
        # Select key features (simplified)
        feature_columns = [
            'sentiment_mean', 'confidence', 'bullish_terms', 'bearish_terms',
            'post_volume', 'Open', 'High', 'Low', 'Volume', 'daily_return'
        ]
        
        # Use only available features
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"   📊 Using {len(available_features)} features")
        
        # Prepare data
        feature_data = data[available_features].values.astype(float)
        target_data = data[target_col].values.astype(float)
        
        # Simple scaling
        from sklearn.preprocessing import MinMaxScaler
        
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        feature_scaled = feature_scaler.fit_transform(feature_data)
        target_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        
        for i in range(lookback, len(feature_scaled)):
            X.append(feature_scaled[i-lookback:i])
            y.append(target_scaled[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"   📦 Sequences: X{X.shape}, y{y.shape}")
        
        return X, y, feature_scaler, target_scaler, available_features
        
    def build_simple_lstm(self, input_shape):
        """🏗️ Costruisce LSTM semplificato"""
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def train_model_safe(self, asset_name, subreddit_filter=None):
        """🚀 Training sicuro del modello"""
        
        print(f"\n🚀 === TRAINING {asset_name.upper()} MODEL ===")
        
        try:
            # Load data with auto-fix
            data = self.load_and_prepare_data(asset_name, subreddit_filter)
            if data is None:
                return None
                
            # Prepare LSTM data
            X, y, feature_scaler, target_scaler, features = self.prepare_simple_lstm_data(data, lookback=3)
            
            if len(X) < 10:
                print(f"❌ Insufficient data for training ({len(X)} samples)")
                return None
                
            # Train-test split (temporal)
            split_idx = max(1, int(0.8 * len(X)))
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Build model
            model = self.build_simple_lstm((X.shape[1], X.shape[2]))
            
            print(f"   🏃 Training model...")
            
            # Train with early stopping
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                epochs=30,
                batch_size=min(8, len(X_train)),
                callbacks=callbacks,
                verbose=0
            )
            
            # Predictions
            if len(X_test) > 0:
                train_pred = model.predict(X_train, verbose=0)
                test_pred = model.predict(X_test, verbose=0)
                
                # Inverse transform
                train_pred = target_scaler.inverse_transform(train_pred)
                test_pred = target_scaler.inverse_transform(test_pred)
                y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
                y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Metrics
                test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
                test_mae = mean_absolute_error(y_test_actual, test_pred)
                test_r2 = r2_score(y_test_actual, test_pred)
                
                # Direction accuracy
                actual_dir = np.sign(np.diff(y_test_actual.flatten()))
                pred_dir = np.sign(np.diff(test_pred.flatten()))
                direction_acc = np.mean(actual_dir == pred_dir) if len(actual_dir) > 0 else 0
                
                print(f"\n📊 === RESULTS ===")
                print(f"   RMSE: ${test_rmse:.2f}")
                print(f"   MAE:  ${test_mae:.2f}")
                print(f"   R²:   {test_r2:.4f}")
                print(f"   Direction Accuracy: {direction_acc:.2%}")
                
                # Simple visualization
                self.create_simple_visualization(
                    y_test_actual, test_pred, 
                    asset_name, subreddit_filter, history
                )
                
                return {
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'r2': test_r2,
                    'direction_accuracy': direction_acc,
                    'data_points': len(X_test)
                }
            else:
                print("❌ No test data available")
                return None
                
        except Exception as e:
            print(f"❌ Error in training: {e}")
            return None
            
    def create_simple_visualization(self, actual, predicted, asset, subreddit, history):
        """📈 Crea visualizzazione semplificata"""
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Predictions plot
            axes[0].plot(actual, label='Actual', alpha=0.8)
            axes[0].plot(predicted, label='Predicted', alpha=0.8)
            axes[0].set_title(f'{asset.upper()} Predictions')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[1].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            os.makedirs('results/lstm_final', exist_ok=True)
            filename = f"{asset}_{subreddit if subreddit else 'overall'}.png"
            plt.savefig(f'results/lstm_final/{filename}', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Plot saved: results/lstm_final/{filename}")
            
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
            
    def run_final_modeling(self):
        """🚀 Esegue modeling finale semplificato"""
        
        print("🚀 === FINAL LSTM MODELING (SIMPLIFIED) ===")
        print("🎯 Quick implementation for thesis\n")
        
        # Priority models
        models = [
            ('bitcoin', 'CryptoCurrency'),
            ('ethereum', 'CryptoCurrency'),
            ('sp500', 'wallstreetbets'),
            ('nasdaq', 'wallstreetbets')
        ]
        
        results = []
        successful_models = 0
        
        for asset, subreddit in models:
            print(f"\n{'='*50}")
            
            result = self.train_model_safe(asset, subreddit)
            
            if result:
                results.append({
                    'asset': asset.upper(),
                    'subreddit': f"r/{subreddit}",
                    **result
                })
                successful_models += 1
                print(f"✅ {asset.upper()} model completed successfully!")
            else:
                print(f"❌ {asset.upper()} model failed")
                
        # Final summary
        print(f"\n🎉 === FINAL SUMMARY ===")
        print(f"✅ Successfully trained: {successful_models}/4 models")
        
        if results:
            print(f"\n📊 MODEL PERFORMANCE:")
            for result in results:
                print(f"🚀 {result['asset']} ({result['subreddit']}):")
                print(f"   📊 RMSE: ${result['rmse']:.2f}")
                print(f"   📈 R²: {result['r2']:.3f}")
                print(f"   🎯 Direction Accuracy: {result['direction_accuracy']:.1%}")
                print(f"   📅 Test samples: {result['data_points']}")
                print()
                
            # Best model
            best = max(results, key=lambda x: x['direction_accuracy'])
            print(f"🏆 BEST MODEL: {best['asset']} with {best['direction_accuracy']:.1%} direction accuracy")
            
            print(f"\n🎓 === READY FOR THESIS ===")
            print(f"📊 {successful_models} working LSTM models")
            print(f"📈 Performance metrics calculated")
            print(f"🎨 Visualizations in results/lstm_final/")
            print(f"✅ Thesis implementation COMPLETED!")
            
        else:
            print("❌ No models trained successfully")
            print("🔧 Check data files manually")
            
        return results

def main():
    """🚀 Main execution"""
    
    predictor = FinancialLSTMWithFix()
    
    try:
        results = predictor.run_final_modeling()
        
        if results:
            print(f"\n🎉 SUCCESS! LSTM models ready for thesis presentation!")
        else:
            print(f"\n❌ No models completed successfully")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()