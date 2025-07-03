import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import json
import logging
from typing import Dict, Tuple

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentFinanceAnalyzer:
    def __init__(self):
        """Inizializza l'analizzatore sentiment-finanza"""
        self.vader = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        logger.info("✅ Sentiment Finance Analyzer inizializzato")
    
    def clean_text(self, text: str) -> str:
        """Pulisce il testo per sentiment analysis"""
        if pd.isna(text):
            return ""
        
        # Rimuovi URL, menzioni, hashtag
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Rimuovi caratteri speciali ma mantieni emoticons importanti
        text = re.sub(r'[^\w\s🚀📈📉💎🌙💰🔥😂😭👍👎❤️💯]', ' ', text)
        
        # Normalizza spazi
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analizza sentiment con VADER"""
        clean_text = self.clean_text(text)
        if not clean_text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        scores = self.vader.polarity_scores(clean_text)
        return scores
    
    def categorize_sentiment(self, compound_score: float) -> str:
        """Categorizza il sentiment in classi"""
        if compound_score >= 0.25:
            return 'bullish'
        elif compound_score <= -0.25:
            return 'bearish'
        else:
            return 'neutral'
    
    def process_reddit_data(self) -> Dict[str, pd.DataFrame]:
        """Processa tutti i dati Reddit con sentiment"""
        logger.info("🔄 Processando dati Reddit per sentiment analysis...")
        
        # Trova file più recenti
        data_dir = "data/raw"
        
        files = {
            'wsb_posts': sorted(glob.glob(f"{data_dir}/wallstreetbets_posts_*.csv"))[-1],
            'wsb_comments': sorted(glob.glob(f"{data_dir}/wallstreetbets_comments_*.csv"))[-1],
            'crypto_posts': sorted(glob.glob(f"{data_dir}/CryptoCurrency_posts_*.csv"))[-1],
            'crypto_comments': sorted(glob.glob(f"{data_dir}/CryptoCurrency_comments_*.csv"))[-1]
        }
        
        results = {}
        
        for data_type, file_path in files.items():
            logger.info(f"📊 Processando {data_type}")
            
            df = pd.read_csv(file_path)
            
            # Colonna testo da analizzare
            if 'posts' in data_type:
                text_col = 'text_content'
            else:  # comments
                text_col = 'comment_body'
            
            # Applica sentiment analysis
            sentiment_scores = df[text_col].apply(self.analyze_sentiment_vader)
            
            # Estrai scores individuali
            df['sentiment_compound'] = [s['compound'] for s in sentiment_scores]
            df['sentiment_pos'] = [s['pos'] for s in sentiment_scores]
            df['sentiment_neu'] = [s['neu'] for s in sentiment_scores]
            df['sentiment_neg'] = [s['neg'] for s in sentiment_scores]
            df['sentiment_category'] = df['sentiment_compound'].apply(self.categorize_sentiment)
            
            # Converti date
            df['datetime'] = pd.to_datetime(df['created_utc'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            
            results[data_type] = df
            logger.info(f"✅ {data_type}: {len(df)} record processati")
        
        return results

    def aggregate_sentiment_temporal(self, reddit_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggrega sentiment per ora e subreddit"""
        logger.info("🕐 Aggregando sentiment per ora...")
        
        # WSB aggregation
        wsb_posts = reddit_data['wsb_posts']
        wsb_comments = reddit_data['wsb_comments']
        
        # Combina posts e comments WSB
        wsb_all = pd.concat([
            wsb_posts[['datetime', 'date', 'hour', 'sentiment_compound', 'sentiment_category']],
            wsb_comments[['datetime', 'date', 'hour', 'sentiment_compound', 'sentiment_category']]
        ])
        
        wsb_hourly = wsb_all.groupby(['date', 'hour']).agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_category': lambda x: (x == 'bullish').sum() / len(x)
        }).reset_index()
        
        wsb_hourly.columns = ['date', 'hour', 'wsb_sentiment_mean', 'wsb_sentiment_std', 
                             'wsb_post_count', 'wsb_bullish_ratio']
        
        # Crypto aggregation
        crypto_posts = reddit_data['crypto_posts']
        crypto_comments = reddit_data['crypto_comments']
        
        crypto_all = pd.concat([
            crypto_posts[['datetime', 'date', 'hour', 'sentiment_compound', 'sentiment_category']],
            crypto_comments[['datetime', 'date', 'hour', 'sentiment_compound', 'sentiment_category']]
        ])
        
        crypto_hourly = crypto_all.groupby(['date', 'hour']).agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_category': lambda x: (x == 'bullish').sum() / len(x)
        }).reset_index()
        
        crypto_hourly.columns = ['date', 'hour', 'crypto_sentiment_mean', 'crypto_sentiment_std',
                                'crypto_post_count', 'crypto_bullish_ratio']
        
        # Merge su date/hour
        hourly_sentiment = wsb_hourly.merge(
            crypto_hourly[['date', 'hour', 'crypto_sentiment_mean', 'crypto_sentiment_std', 
                          'crypto_post_count', 'crypto_bullish_ratio']], 
            on=['date', 'hour'], 
            how='outer'
        )
        
        # Fill NaN con valori neutri
        sentiment_cols = ['wsb_sentiment_mean', 'crypto_sentiment_mean']
        hourly_sentiment[sentiment_cols] = hourly_sentiment[sentiment_cols].fillna(0)
        
        count_cols = ['wsb_post_count', 'crypto_post_count']
        hourly_sentiment[count_cols] = hourly_sentiment[count_cols].fillna(0)
        
        ratio_cols = ['wsb_bullish_ratio', 'crypto_bullish_ratio']
        hourly_sentiment[ratio_cols] = hourly_sentiment[ratio_cols].fillna(0.5)
        
        logger.info(f"✅ Sentiment aggregato: {len(hourly_sentiment)} ore")
        return hourly_sentiment
    
    def load_financial_data(self) -> Dict[str, pd.DataFrame]:
        """Carica dati finanziari"""
        logger.info("💰 Caricando dati finanziari...")
        
        data_dir = "data/financial"
        
        files = {
            'sp500': sorted(glob.glob(f"{data_dir}/sp500_hourly_*.csv"))[-1],
            'bitcoin': sorted(glob.glob(f"{data_dir}/bitcoin_hourly_*.csv"))[-1],
            'ethereum': sorted(glob.glob(f"{data_dir}/ethereum_hourly_*.csv"))[-1],
            'nasdaq': sorted(glob.glob(f"{data_dir}/nasdaq_hourly_*.csv"))[-1]
        }
        
        financial_data = {}
        
        for asset, file_path in files.items():
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['Datetime'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            
            # Target per ML (movimento prezzo nell'ora successiva)
            df['future_price'] = df['Close'].shift(-1)
            df['future_return'] = (df['future_price'] - df['Close']) / df['Close']
            df['price_direction'] = (df['future_return'] > 0).astype(int)
            
            financial_data[asset] = df
            logger.info(f"📈 {asset}: {len(df)} ore caricate")
        
        return financial_data
    
    def create_ml_dataset(self, sentiment_df: pd.DataFrame, financial_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Crea dataset per machine learning"""
        logger.info("🤖 Creando dataset per ML...")
        
        datasets = []
        
        # WSB → S&P 500 / NASDAQ
        for market in ['sp500', 'nasdaq']:
            market_df = financial_data[market][['date', 'hour', 'Close', 'price_change', 
                                             'future_return', 'price_direction', 'volatility']].copy()
            
            merged = market_df.merge(sentiment_df, on=['date', 'hour'], how='inner')
            merged['target_asset'] = market
            merged['sentiment_score'] = merged['wsb_sentiment_mean']
            merged['post_volume'] = merged['wsb_post_count']
            merged['bullish_ratio'] = merged['wsb_bullish_ratio']
            
            datasets.append(merged)
        
        # Crypto → Bitcoin / Ethereum  
        for crypto in ['bitcoin', 'ethereum']:
            crypto_df = financial_data[crypto][['date', 'hour', 'Close', 'price_change',
                                              'future_return', 'price_direction', 'volatility']].copy()
            
            merged = crypto_df.merge(sentiment_df, on=['date', 'hour'], how='inner')
            merged['target_asset'] = crypto
            merged['sentiment_score'] = merged['crypto_sentiment_mean']
            merged['post_volume'] = merged['crypto_post_count']
            merged['bullish_ratio'] = merged['crypto_bullish_ratio']
            
            datasets.append(merged)
        
        # Combina tutto
        ml_dataset = pd.concat(datasets, ignore_index=True)
        
        # Features aggiuntive
        ml_dataset['sentiment_strength'] = np.abs(ml_dataset['sentiment_score'])
        ml_dataset['hour_sin'] = np.sin(2 * np.pi * ml_dataset['hour'] / 24)
        ml_dataset['hour_cos'] = np.cos(2 * np.pi * ml_dataset['hour'] / 24)
        
        ml_dataset = ml_dataset.dropna()
        
        logger.info(f"🎯 Dataset ML creato: {len(ml_dataset)} sample")
        return ml_dataset
    
    def train_baseline_models(self, ml_dataset: pd.DataFrame) -> Dict:
        """Addestra modelli baseline"""
        logger.info("🎯 Addestrando modelli baseline...")
        
        feature_cols = ['sentiment_score', 'sentiment_strength', 'post_volume', 'bullish_ratio',
                       'price_change', 'volatility', 'hour_sin', 'hour_cos']
        
        X = ml_dataset[feature_cols]
        y = ml_dataset['price_direction']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"🔄 Training {name}...")
            
            if 'Logistic' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"✅ {name}: Accuracy={accuracy:.3f}, AUC={auc_score:.3f}")
        
        return results

def main():
    """Funzione principale"""
    logger.info("🚀 === SENTIMENT ANALYSIS + ML BASELINE ===")
    
    analyzer = SentimentFinanceAnalyzer()
    
    try:
        # 1. Processa dati Reddit
        reddit_data = analyzer.process_reddit_data()
        
        # 2. Aggrega sentiment per ora
        hourly_sentiment = analyzer.aggregate_sentiment_temporal(reddit_data)
        
        # 3. Carica dati finanziari
        financial_data = analyzer.load_financial_data()
        
        # 4. Crea dataset ML
        ml_dataset = analyzer.create_ml_dataset(hourly_sentiment, financial_data)
        
        # 5. Addestra modelli
        model_results = analyzer.train_baseline_models(ml_dataset)
        
        # 6. Risultati
        print("\n📊 === RISULTATI MODELLI BASELINE ===")
        for name, results in model_results.items():
            print(f"\n🎯 {name}:")
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   AUC Score: {results['auc']:.3f}")
        
        # 7. Sentiment per asset
        print("\n💰 === SENTIMENT PER ASSET ===")
        for asset in ml_dataset['target_asset'].unique():
            asset_data = ml_dataset[ml_dataset['target_asset'] == asset]
            avg_sentiment = asset_data['sentiment_score'].mean()
            correlation = asset_data['sentiment_score'].corr(asset_data['future_return'])
            
            print(f"\n📈 {asset.upper()}:")
            print(f"   Sentiment medio: {avg_sentiment:.3f}")
            print(f"   Correlazione sentiment-return: {correlation:.3f}")
        
        # Salva risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ml_dataset.to_csv(f"data/processed/ml_dataset_{timestamp}.csv", index=False)
        
        print(f"\n🎉 === ANALISI COMPLETATA ===")
        
    except Exception as e:
        logger.error(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()