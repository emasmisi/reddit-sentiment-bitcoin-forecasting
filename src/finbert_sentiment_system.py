#!/usr/bin/env python3
"""
🧠 FINBERT ADVANCED SENTIMENT ANALYSIS SYSTEM - FIXED VERSION
================================================================
Sistema di sentiment analysis CORRETTO per il tuo database schema
- Auto-detect schema columns
- Handle posts vs comments differences
- FinBERT + VADER Enhanced + Ensemble
"""

import pandas as pd
import numpy as np
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinBERTAdvancedSentimentAnalyzer:
    """🧠 Analizzatore sentiment avanzato con FinBERT + VADER Enhanced"""
    
    def __init__(self, db_path: str = "data/reddit_data.db"):
        self.db_path = db_path
        self.setup_models()
        self.setup_financial_dictionaries()
        
    def setup_models(self):
        """🔥 Setup FinBERT e VADER Enhanced"""
        logger.info("🚀 Caricando FinBERT modello avanzato...")
        
        try:
            # FinBERT - Modello specializzato per finanza
            self.finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ FinBERT caricato con successo!")
            
        except Exception as e:
            logger.warning(f"⚠️ FinBERT non disponibile, usando backup: {e}")
            # Backup: usa BERT base
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
        
        # VADER Enhanced
        self.vader = SentimentIntensityAnalyzer()
        logger.info("✅ VADER Enhanced configurato!")
        
    def setup_financial_dictionaries(self):
        """💰 Setup dizionari finanziari e patterns"""
        
        # WSB Slang & Financial Terms
        self.bullish_terms = {
            # WSB Slang
            'moon', 'mooning', 'to the moon', 'rocket', 'diamond hands', 'hodl', 'hold',
            'buy the dip', 'tendies', 'stonks', 'ape', 'apes strong', 'this is the way',
            
            # Financial Bullish
            'bullish', 'buy', 'long', 'pump', 'breakout', 'rally', 'surge', 'squeeze',
            'gains', 'profit', 'green', 'up', 'rise', 'increase', 'growth', 'strong',
            'outperform', 'beat estimates', 'record high', 'all time high', 'ath',
            
            # Crypto specific
            'lambo', 'when lambo', 'btfd', 'dca', 'stacking'
        }
        
        self.bearish_terms = {
            # WSB Bearish
            'paper hands', 'sell', 'dump', 'crash', 'rekt', 'bag holder', 'bagholder',
            'fud', 'fear', 'panic', 'rug pull', 'exit scam',
            
            # Financial Bearish  
            'bearish', 'short', 'put', 'decline', 'fall', 'drop', 'red', 'down',
            'loss', 'losses', 'miss estimates', 'disappointing', 'weak', 'struggle',
            'underperform', 'correction', 'bear market', 'recession', 'bubble'
        }
        
        # Emoji patterns finanziari
        self.bullish_emojis = ['🚀', '📈', '💎', '🌙', '💰', '🤑', '🔥', '💪', '👍', '✅', '🎯']
        self.bearish_emojis = ['📉', '💩', '😭', '😢', '🔻', '⬇️', '❌', '💸', '🐻', '👎']
        
        # Ticker patterns
        self.ticker_pattern = r'\$[A-Z]{1,5}\b'
        
        logger.info("✅ Dizionari finanziari configurati!")
        
    def extract_financial_features(self, text: str) -> Dict[str, float]:
        """💡 Estrazione features finanziarie avanzate"""
        
        text_lower = text.lower()
        features = {}
        
        # Ticker mentions
        tickers = re.findall(self.ticker_pattern, text, re.IGNORECASE)
        features['ticker_count'] = len(tickers)
        features['has_ticker'] = 1.0 if tickers else 0.0
        
        # Bullish/Bearish term counts
        bullish_count = sum(1 for term in self.bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in self.bearish_terms if term in text_lower)
        
        features['bullish_terms'] = bullish_count
        features['bearish_terms'] = bearish_count
        features['sentiment_ratio'] = (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
        
        # Emoji analysis
        bullish_emoji_count = sum(1 for emoji in self.bullish_emojis if emoji in text)
        bearish_emoji_count = sum(1 for emoji in self.bearish_emojis if emoji in text)
        
        features['bullish_emojis'] = bullish_emoji_count
        features['bearish_emojis'] = bearish_emoji_count
        features['emoji_sentiment'] = (bullish_emoji_count - bearish_emoji_count) / max(1, bullish_emoji_count + bearish_emoji_count)
        
        # Text characteristics
        features['text_length'] = len(text)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
        
    def analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """🧠 Analisi sentiment con FinBERT"""
        
        try:
            # Preprocessing per FinBERT
            clean_text = self.preprocess_text(text)
            
            if len(clean_text.strip()) < 5:
                return {'finbert_label': 'neutral', 'finbert_score': 0.0, 'finbert_confidence': 0.0}
                
            # FinBERT prediction
            result = self.finbert_pipeline(clean_text[:512])  # Max length limit
            
            # Parse risultati (formato diverso per modelli diversi)
            if isinstance(result, list):
                result = result[0]
                
            label = result.get('label', 'NEUTRAL')
            score = result.get('score', 0.0)
            
            # Normalizza labels
            if 'LABEL_0' in label or 'neutral' in label.lower():
                sentiment_label = 'neutral'
                sentiment_score = 0.0
            elif 'LABEL_1' in label or 'positive' in label.lower():
                sentiment_label = 'bullish'
                sentiment_score = score
            elif 'LABEL_2' in label or 'negative' in label.lower():
                sentiment_label = 'bearish'
                sentiment_score = -score
            else:
                sentiment_label = 'neutral'
                sentiment_score = 0.0
                
            return {
                'finbert_label': sentiment_label,
                'finbert_score': sentiment_score,
                'finbert_confidence': score
            }
            
        except Exception as e:
            logger.warning(f"⚠️ FinBERT error: {e}")
            return {'finbert_label': 'neutral', 'finbert_score': 0.0, 'finbert_confidence': 0.0}
            
    def analyze_with_vader_enhanced(self, text: str) -> Dict[str, float]:
        """⚡ VADER Enhanced con context finanziario"""
        
        # Basic VADER
        vader_scores = self.vader.polarity_scores(text)
        
        # Enhancement con context finanziario
        financial_features = self.extract_financial_features(text)
        
        # Boost/penalizza basato su financial features
        enhancement_factor = 0.0
        
        # Ticker mentions aumentano l'importanza
        if financial_features['has_ticker']:
            enhancement_factor += 0.1
            
        # Financial terms boost
        if financial_features['sentiment_ratio'] > 0.5:
            enhancement_factor += 0.2
        elif financial_features['sentiment_ratio'] < -0.5:
            enhancement_factor -= 0.2
            
        # Emoji boost
        if financial_features['emoji_sentiment'] > 0.5:
            enhancement_factor += 0.15
        elif financial_features['emoji_sentiment'] < -0.5:
            enhancement_factor -= 0.15
            
        # Apply enhancement
        enhanced_compound = np.clip(
            vader_scores['compound'] + enhancement_factor,
            -1.0, 1.0
        )
        
        # Categorize
        if enhanced_compound >= 0.25:
            vader_label = 'bullish'
        elif enhanced_compound <= -0.25:
            vader_label = 'bearish'
        else:
            vader_label = 'neutral'
            
        return {
            'vader_compound': enhanced_compound,
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'vader_label': vader_label,
            'vader_enhancement': enhancement_factor
        }
        
    def ensemble_prediction(self, finbert_result: Dict, vader_result: Dict, 
                          financial_features: Dict) -> Dict[str, float]:
        """🎯 Ensemble method per prediction finale"""
        
        # Weights basati su confidence e features
        finbert_weight = 0.7  # FinBERT ha peso maggiore
        vader_weight = 0.3
        
        # Adjust weights basato su financial features
        if financial_features['has_ticker'] or financial_features['bullish_terms'] > 0 or financial_features['bearish_terms'] > 0:
            finbert_weight = 0.8  # Più peso a FinBERT per testo finanziario
            vader_weight = 0.2
            
        # Ensemble score
        ensemble_score = (
            finbert_result['finbert_score'] * finbert_weight +
            vader_result['vader_compound'] * vader_weight
        )
        
        # Final label
        if ensemble_score >= 0.25:
            final_label = 'bullish'
        elif ensemble_score <= -0.25:
            final_label = 'bearish'
        else:
            final_label = 'neutral'
            
        # Agreement between models
        agreement = (finbert_result['finbert_label'] == vader_result['vader_label'])
        
        # Confidence based on agreement and scores
        confidence = (
            finbert_result['finbert_confidence'] * finbert_weight +
            abs(vader_result['vader_compound']) * vader_weight
        )
        
        if agreement:
            confidence *= 1.2  # Boost confidence se models agree
            
        confidence = min(confidence, 1.0)
        
        return {
            'final_label': final_label,
            'final_score': ensemble_score,
            'confidence': confidence,
            'model_agreement': agreement,
            'finbert_weight': finbert_weight,
            'vader_weight': vader_weight
        }
        
    def preprocess_text(self, text: str) -> str:
        """🧹 Preprocessing intelligente del testo"""
        
        if pd.isna(text) or not text:
            return ""
            
        # Basic cleaning
        text = str(text).strip()
        
        # Remove URLs but keep ticker mentions
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
        
    def get_database_schema(self) -> Dict[str, List[str]]:
        """📋 Rileva automaticamente lo schema del database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get posts columns
        cursor.execute("PRAGMA table_info(posts)")
        posts_columns = [row[1] for row in cursor.fetchall()]
        
        # Get comments columns  
        cursor.execute("PRAGMA table_info(comments)")
        comments_columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        schema = {
            'posts': posts_columns,
            'comments': comments_columns
        }
        
        logger.info(f"📋 Schema rilevato:")
        logger.info(f"   Posts: {len(posts_columns)} colonne")
        logger.info(f"   Comments: {len(comments_columns)} colonne")
        
        return schema
        
    def process_reddit_data(self) -> pd.DataFrame:
        """🔄 Processa tutto il dataset Reddit con FinBERT - SCHEMA ADAPTIVE"""
        
        logger.info("🚀 Inizio processing con FinBERT Advanced (Schema Adaptive)...")
        
        # Rileva schema database
        schema = self.get_database_schema()
        
        # Connetti al database
        conn = sqlite3.connect(self.db_path)
        
        # Carica posts e comments
        posts_df = pd.read_sql_query("SELECT * FROM posts", conn)
        comments_df = pd.read_sql_query("SELECT * FROM comments", conn)
        
        logger.info(f"📊 Caricati {len(posts_df)} posts e {len(comments_df)} comments")
        
        # Combina per processing - SCHEMA ADAPTIVE
        all_data = []
        
        # Process posts
        for _, post in posts_df.iterrows():
            # Text combination
            text_parts = []
            if 'title' in schema['posts'] and pd.notna(post.get('title')):
                text_parts.append(str(post['title']))
            if 'selftext' in schema['posts'] and pd.notna(post.get('selftext')):
                text_parts.append(str(post['selftext']))
            if 'text_content' in schema['posts'] and pd.notna(post.get('text_content')):
                text_parts.append(str(post['text_content']))
                
            text = ' '.join(text_parts).strip()
            
            # Skip if no text
            if not text or len(text) < 5:
                continue
                
            all_data.append({
                'id': post.get('post_id', f"post_{len(all_data)}"),
                'type': 'post',
                'subreddit': post.get('subreddit', 'unknown'),
                'text': text,
                'created_date': post.get('created_date', ''),
                'score': post.get('score', 0),
                'created_hour': post.get('created_hour', 0) if 'created_hour' in schema['posts'] else 0
            })
            
        # Process comments  
        for _, comment in comments_df.iterrows():
            text = str(comment.get('comment_body', ''))
            
            # Skip if no text
            if not text or len(text.strip()) < 5 or text in ['[deleted]', '[removed]']:
                continue
                
            all_data.append({
                'id': comment.get('comment_id', f"comment_{len(all_data)}"),
                'type': 'comment', 
                'subreddit': comment.get('subreddit', 'unknown'),
                'text': text,
                'created_date': comment.get('created_date', ''),
                'score': comment.get('score', 0),
                'created_hour': 0  # Comments don't have created_hour in your schema
            })
            
        df = pd.DataFrame(all_data)
        logger.info(f"🎯 Dataset combinato: {len(df)} righe totali")
        
        # Apply sentiment analysis
        results = []
        
        for i, row in df.iterrows():
            if i % 100 == 0:
                logger.info(f"📈 Processati {i}/{len(df)} ({i/len(df)*100:.1f}%)")
                
            text = row['text']
            
            try:
                # Extract financial features
                financial_features = self.extract_financial_features(text)
                
                # FinBERT analysis
                finbert_result = self.analyze_with_finbert(text)
                
                # VADER Enhanced analysis
                vader_result = self.analyze_with_vader_enhanced(text)
                
                # Ensemble prediction
                ensemble_result = self.ensemble_prediction(
                    finbert_result, vader_result, financial_features
                )
                
                # Combine all results
                result = {
                    'id': row['id'],
                    'type': row['type'],
                    'subreddit': row['subreddit'],
                    'text': text[:200],  # Truncate for storage
                    'created_date': row['created_date'],
                    'score': row['score'],
                    'created_hour': row['created_hour'],
                    
                    # Financial features
                    **financial_features,
                    
                    # FinBERT results
                    **finbert_result,
                    
                    # VADER Enhanced results
                    **vader_result,
                    
                    # Ensemble results
                    **ensemble_result,
                    
                    # Metadata
                    'processed_timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing text {i}: {e}")
                continue
                
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/processed/finbert_sentiment_analysis_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Sentiment analysis completata!")
        logger.info(f"💾 Risultati salvati: {output_file}")
        logger.info(f"📊 Processati: {len(results_df)} record")
        
        conn.close()
        
        return results_df
        
    def generate_summary_stats(self, results_df: pd.DataFrame) -> Dict:
        """📈 Genera statistiche riassuntive"""
        
        stats = {
            'total_records': len(results_df),
            'by_subreddit': results_df['subreddit'].value_counts().to_dict(),
            'by_sentiment_finbert': results_df['finbert_label'].value_counts().to_dict(),
            'by_sentiment_vader': results_df['vader_label'].value_counts().to_dict(),
            'by_sentiment_ensemble': results_df['final_label'].value_counts().to_dict(),
            'model_agreement_rate': results_df['model_agreement'].mean(),
            'avg_confidence': results_df['confidence'].mean(),
            'financial_features': {
                'with_tickers': results_df['has_ticker'].sum(),
                'avg_bullish_terms': results_df['bullish_terms'].mean(),
                'avg_bearish_terms': results_df['bearish_terms'].mean(),
                'avg_sentiment_ratio': results_df['sentiment_ratio'].mean()
            }
        }
        
        return stats

def main():
    """🚀 Main execution"""
    
    logger.info("🧠 === FINBERT ADVANCED SENTIMENT ANALYSIS ===")
    
    try:
        # Initialize analyzer
        analyzer = FinBERTAdvancedSentimentAnalyzer()
        
        # Process data
        results_df = analyzer.process_reddit_data()
        
        # Generate stats
        stats = analyzer.generate_summary_stats(results_df)
        
        # Print results
        print("\n🎉 === RISULTATI FINBERT SENTIMENT ANALYSIS ===")
        print(f"✅ Record totali processati: {stats['total_records']:,}")
        print(f"📊 Agreement rate tra modelli: {stats['model_agreement_rate']:.1%}")
        print(f"🎯 Confidence media: {stats['avg_confidence']:.3f}")
        
        print(f"\n💰 FEATURES FINANZIARIE:")
        print(f"   Posts/comments con ticker: {stats['financial_features']['with_tickers']:,}")
        print(f"   Media termini bullish: {stats['financial_features']['avg_bullish_terms']:.2f}")
        print(f"   Media termini bearish: {stats['financial_features']['avg_bearish_terms']:.2f}")
        
        print(f"\n🧠 SENTIMENT DISTRIBUTION (FinBERT):")
        for sentiment, count in stats['by_sentiment_finbert'].items():
            print(f"   {sentiment}: {count:,} ({count/stats['total_records']*100:.1f}%)")
            
        print(f"\n🎯 SENTIMENT DISTRIBUTION (Ensemble):")
        for sentiment, count in stats['by_sentiment_ensemble'].items():
            print(f"   {sentiment}: {count:,} ({count/stats['total_records']*100:.1f}%)")
            
        print(f"\n📈 PER SUBREDDIT:")
        for subreddit, count in stats['by_subreddit'].items():
            print(f"   r/{subreddit}: {count:,}")
            
        print(f"\n🚀 SENTIMENT ANALYSIS CON FINBERT COMPLETATA! 🚀")
        print(f"I tuoi dati sono ora pronti per l'analisi predittiva avanzata! 💎")
        
    except Exception as e:
        logger.error(f"❌ Errore durante l'esecuzione: {e}")
        raise

if __name__ == "__main__":
    main()