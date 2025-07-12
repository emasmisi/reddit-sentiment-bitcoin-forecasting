#!/usr/bin/env python3
"""
📊 COMPREHENSIVE CORRELATION ANALYSIS SYSTEM
=============================================
FASE 1: Analisi correlazioni sentiment-financial per validazione dataset
- Statistical significance testing
- Lead-lag relationship analysis  
- Asset-specific correlation discovery
- Economic interpretation e insights generation
- Foundation per PHASE 2 (LSTM modeling)
"""

import pandas as pd
import numpy as np
import sqlite3
import glob
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ComprehensiveCorrelationAnalyzer:
    """📊 Sistema completo per analisi correlazioni sentiment-financial"""
    
    def __init__(self):
        print("📊 === COMPREHENSIVE CORRELATION ANALYSIS ===")
        print("🎯 FASE 1: Validazione scientifica del dataset\n")
        
        self.sentiment_data = None
        self.financial_data = {}
        self.correlation_results = {}
        self.statistical_tests = {}
        self.lead_lag_results = {}
        
        # Setup visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_sentiment_data(self):
        """📊 Carica e prepara dati sentiment FinBERT"""
        
        print("📊 Loading FinBERT sentiment data...")
        
        # Trova file sentiment più recente
        sentiment_files = glob.glob("data/processed/finbert_sentiment_analysis_*.csv")
        
        if not sentiment_files:
            print("❌ No FinBERT sentiment files found!")
            return False
            
        latest_file = sorted(sentiment_files)[-1]
        print(f"📁 Loading: {os.path.basename(latest_file)}")
        
        try:
            df = pd.read_csv(latest_file)
            
            # Data conversion and cleaning
            df['created_date'] = pd.to_datetime(df['created_date'])
            df = df.dropna(subset=['final_score', 'created_date'])
            
            # Aggregate by date and subreddit for daily analysis
            daily_sentiment = df.groupby(['created_date', 'subreddit']).agg({
                # Core sentiment metrics
                'final_score': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'finbert_score': 'mean',
                'vader_compound': 'mean',
                
                # Financial features
                'bullish_terms': 'sum',
                'bearish_terms': 'sum', 
                'has_ticker': 'sum',
                'ticker_count': 'sum',
                
                # Volume metrics
                'id': 'count'
            }).reset_index()
            
            # Flatten multi-level columns
            daily_sentiment.columns = [
                'created_date', 'subreddit', 'sentiment_mean', 'sentiment_std', 'sentiment_count',
                'confidence', 'finbert_score', 'vader_compound', 'bullish_terms', 'bearish_terms',
                'ticker_mentions', 'ticker_count', 'post_volume'
            ]
            
            # Additional derived metrics
            daily_sentiment['sentiment_momentum'] = daily_sentiment.groupby('subreddit')['sentiment_mean'].diff()
            daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_std'].fillna(0)
            daily_sentiment['bullish_ratio'] = (
                daily_sentiment['bullish_terms'] / 
                (daily_sentiment['bullish_terms'] + daily_sentiment['bearish_terms'] + 1)
            )
            
            # Overall market sentiment (aggregated across subreddits)
            overall_sentiment = df.groupby('created_date').agg({
                'final_score': ['mean', 'std'],
                'confidence': 'mean',
                'finbert_score': 'mean',
                'bullish_terms': 'sum',
                'bearish_terms': 'sum',
                'has_ticker': 'sum',
                'id': 'count'
            }).reset_index()
            
            overall_sentiment.columns = [
                'created_date', 'overall_sentiment_mean', 'overall_sentiment_std',
                'overall_confidence', 'overall_finbert', 'overall_bullish',
                'overall_bearish', 'overall_tickers', 'overall_volume'
            ]
            
            overall_sentiment['overall_momentum'] = overall_sentiment['overall_sentiment_mean'].diff()
            overall_sentiment['overall_bullish_ratio'] = (
                overall_sentiment['overall_bullish'] / 
                (overall_sentiment['overall_bullish'] + overall_sentiment['overall_bearish'] + 1)
            )
            
            self.sentiment_data = {
                'daily_by_subreddit': daily_sentiment,
                'daily_overall': overall_sentiment,
                'raw_data': df
            }
            
            print(f"✅ Sentiment data loaded successfully!")
            print(f"   📅 Period: {daily_sentiment['created_date'].min().date()} → {daily_sentiment['created_date'].max().date()}")
            print(f"   📊 Daily records: {len(daily_sentiment)} (by subreddit)")
            print(f"   📈 Overall daily records: {len(overall_sentiment)}")
            print(f"   🏷️ Subreddits: {daily_sentiment['subreddit'].unique()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading sentiment data: {e}")
            return False
            
    def load_financial_data(self):
        """📈 Carica e prepara dati finanziari"""
        
        print("\n📈 Loading financial market data...")
        
        financial_files = glob.glob("data/financial/*.csv")
        
        if not financial_files:
            print("❌ No financial data files found!")
            return False
            
        for file_path in financial_files:
            filename = os.path.basename(file_path)
            asset_name = filename.split('_')[0]
            
            try:
                df = pd.read_csv(file_path)
                
                # DateTime conversion
                df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
                df['Date'] = df['Datetime'].dt.date
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Daily aggregation
                daily_financial = df.groupby('Date').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                    'price_change': 'sum'
                }).reset_index()
                
                # Calculate financial metrics
                daily_financial['daily_return'] = (
                    (daily_financial['Close'] - daily_financial['Open']) / daily_financial['Open']
                )
                daily_financial['daily_range'] = (
                    (daily_financial['High'] - daily_financial['Low']) / daily_financial['Open']
                )
                daily_financial['is_positive_day'] = (daily_financial['daily_return'] > 0).astype(int)
                
                # Volatility measures
                daily_financial['return_abs'] = daily_financial['daily_return'].abs()
                daily_financial['volatility_3d'] = daily_financial['return_abs'].rolling(3).mean()
                daily_financial['volatility_7d'] = daily_financial['return_abs'].rolling(7).mean()
                
                # Momentum indicators
                daily_financial['momentum_1d'] = daily_financial['daily_return']
                daily_financial['momentum_3d'] = daily_financial['Close'].pct_change(3)
                daily_financial['momentum_7d'] = daily_financial['Close'].pct_change(7)
                
                # Price level features
                daily_financial['price_ma_7d'] = daily_financial['Close'].rolling(7).mean()
                daily_financial['price_distance_ma'] = (
                    (daily_financial['Close'] - daily_financial['price_ma_7d']) / daily_financial['price_ma_7d']
                )
                
                self.financial_data[asset_name] = daily_financial
                
                print(f"✅ {asset_name.upper()}: {len(daily_financial)} daily records")
                print(f"   📅 Period: {daily_financial['Date'].min().date()} → {daily_financial['Date'].max().date()}")
                
            except Exception as e:
                print(f"❌ Error loading {asset_name}: {e}")
                
        print(f"\n📊 Loaded {len(self.financial_data)} financial assets")
        return len(self.financial_data) > 0
        
    def analyze_asset_correlations(self, asset_name, subreddit_filter=None):
        """🔍 Analizza correlazioni per un asset specifico"""
        
        if asset_name not in self.financial_data:
            print(f"❌ Asset {asset_name} not found in financial data")
            return None
            
        print(f"\n🔍 === CORRELATION ANALYSIS: {asset_name.upper()} ===")
        
        # Prepare sentiment data
        if subreddit_filter:
            sentiment_df = self.sentiment_data['daily_by_subreddit'][
                self.sentiment_data['daily_by_subreddit']['subreddit'] == subreddit_filter
            ].copy()
            print(f"📊 Using sentiment from: r/{subreddit_filter}")
        else:
            sentiment_df = self.sentiment_data['daily_overall'].copy()
            print(f"📊 Using overall aggregated sentiment")
            
        # Merge with financial data
        financial_df = self.financial_data[asset_name].copy()
        
        if subreddit_filter:
            merged = sentiment_df.merge(
                financial_df,
                left_on='created_date',
                right_on='Date',
                how='inner'
            )
        else:
            merged = sentiment_df.merge(
                financial_df,
                left_on='created_date',
                right_on='Date',
                how='inner'
            )
            
        if len(merged) == 0:
            print(f"❌ No overlapping data for {asset_name}")
            return None
            
        print(f"📅 Analysis period: {len(merged)} days ({merged['Date'].min().date()} → {merged['Date'].max().date()})")
        
        # Define correlation pairs
        if subreddit_filter:
            sentiment_cols = [
                'sentiment_mean', 'confidence', 'finbert_score', 'vader_compound',
                'bullish_terms', 'bearish_terms', 'bullish_ratio', 'post_volume',
                'sentiment_momentum', 'sentiment_volatility'
            ]
        else:
            sentiment_cols = [
                'overall_sentiment_mean', 'overall_confidence', 'overall_finbert',
                'overall_bullish', 'overall_bearish', 'overall_bullish_ratio',
                'overall_volume', 'overall_momentum'
            ]
            
        financial_cols = [
            'daily_return', 'return_abs', 'daily_range', 'volatility_3d', 'volatility_7d',
            'momentum_1d', 'momentum_3d', 'momentum_7d', 'price_distance_ma'
        ]
        
        # Calculate correlations
        correlations = {}
        
        for sent_col in sentiment_cols:
            if sent_col not in merged.columns:
                continue
                
            for fin_col in financial_cols:
                if fin_col not in merged.columns:
                    continue
                    
                # Remove NaN values
                valid_data = merged[[sent_col, fin_col]].dropna()
                
                if len(valid_data) < 10:  # Need minimum data
                    continue
                    
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(valid_data[sent_col], valid_data[fin_col])
                
                # Spearman correlation (rank-based, more robust)
                spearman_corr, spearman_p = spearmanr(valid_data[sent_col], valid_data[fin_col])
                
                # Store if significant or strong
                if abs(pearson_corr) > 0.1 or pearson_p < 0.1:
                    correlations[f"{sent_col}_vs_{fin_col}"] = {
                        'pearson_corr': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_p': spearman_p,
                        'n_observations': len(valid_data),
                        'significance': 'Significant' if pearson_p < 0.05 else 'Marginally Significant' if pearson_p < 0.1 else 'Not Significant'
                    }
        
        # Sort by absolute correlation strength
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]['pearson_corr']),
            reverse=True
        )
        
        # Display top correlations
        print(f"\n🎯 TOP CORRELATIONS FOR {asset_name.upper()}:")
        print("-" * 80)
        
        significant_count = 0
        
        for i, (corr_name, stats_data) in enumerate(sorted_correlations[:10], 1):
            pearson_corr = stats_data['pearson_corr']
            pearson_p = stats_data['pearson_p']
            significance = stats_data['significance']
            n_obs = stats_data['n_observations']
            
            # Direction and strength
            direction = "📈 Positive" if pearson_corr > 0 else "📉 Negative"
            
            if abs(pearson_corr) > 0.5:
                strength = "STRONG"
            elif abs(pearson_corr) > 0.3:
                strength = "MODERATE"
            elif abs(pearson_corr) > 0.15:
                strength = "WEAK-MODERATE"
            else:
                strength = "WEAK"
                
            # Significance indicators
            if pearson_p < 0.001:
                sig_indicator = "***"
            elif pearson_p < 0.01:
                sig_indicator = "**"
            elif pearson_p < 0.05:
                sig_indicator = "*"
            else:
                sig_indicator = ""
                
            print(f"{i:2d}. {corr_name}")
            print(f"    {direction} | {strength} | r={pearson_corr:+.3f}{sig_indicator} | p={pearson_p:.4f} | n={n_obs}")
            
            if pearson_p < 0.05:
                significant_count += 1
                
        print(f"\n📊 SUMMARY:")
        print(f"   🎯 Significant correlations (p<0.05): {significant_count}")
        print(f"   📈 Total correlations analyzed: {len(correlations)}")
        print(f"   📅 Data points: {len(merged)} days")
        
        # Store results
        result_key = f"{asset_name}_{subreddit_filter if subreddit_filter else 'overall'}"
        self.correlation_results[result_key] = {
            'correlations': correlations,
            'sorted_correlations': sorted_correlations,
            'merged_data': merged,
            'significant_count': significant_count,
            'total_correlations': len(correlations),
            'data_points': len(merged)
        }
        
        return self.correlation_results[result_key]
        
    def analyze_lead_lag_relationships(self, asset_name, subreddit_filter=None, max_lags=5):
        """⏰ Analizza relazioni lead-lag tra sentiment e prezzi"""
        
        print(f"\n⏰ === LEAD-LAG ANALYSIS: {asset_name.upper()} ===")
        
        result_key = f"{asset_name}_{subreddit_filter if subreddit_filter else 'overall'}"
        
        if result_key not in self.correlation_results:
            print(f"❌ No correlation results found. Run analyze_asset_correlations first.")
            return None
            
        merged = self.correlation_results[result_key]['merged_data'].copy()
        
        # Select key variables for lead-lag analysis
        if subreddit_filter:
            sentiment_var = 'sentiment_mean'
        else:
            sentiment_var = 'overall_sentiment_mean'
            
        financial_vars = ['daily_return', 'return_abs', 'volatility_3d']
        
        lead_lag_results = {}
        
        for fin_var in financial_vars:
            if fin_var not in merged.columns:
                continue
                
            print(f"\n📊 {sentiment_var} vs {fin_var}:")
            
            lag_correlations = {}
            
            # Test different lags
            for lag in range(-max_lags, max_lags + 1):
                try:
                    if lag == 0:
                        # Contemporaneous correlation
                        corr_data = merged[[sentiment_var, fin_var]].dropna()
                        corr, p_val = pearsonr(corr_data[sentiment_var], corr_data[fin_var])
                        lag_label = "Contemporaneous"
                    elif lag > 0:
                        # Sentiment leads financial (sentiment[t] vs financial[t+lag])
                        shifted_financial = merged[fin_var].shift(-lag)
                        corr_data = pd.DataFrame({
                            'sentiment': merged[sentiment_var],
                            'financial': shifted_financial
                        }).dropna()
                        
                        if len(corr_data) < 10:
                            continue
                            
                        corr, p_val = pearsonr(corr_data['sentiment'], corr_data['financial'])
                        lag_label = f"Sentiment leads by {lag} days"
                    else:
                        # Financial leads sentiment (financial[t] vs sentiment[t+|lag|])
                        shifted_sentiment = merged[sentiment_var].shift(-abs(lag))
                        corr_data = pd.DataFrame({
                            'financial': merged[fin_var],
                            'sentiment': shifted_sentiment
                        }).dropna()
                        
                        if len(corr_data) < 10:
                            continue
                            
                        corr, p_val = pearsonr(corr_data['financial'], corr_data['sentiment'])
                        lag_label = f"Financial leads by {abs(lag)} days"
                        
                    lag_correlations[lag] = {
                        'correlation': corr,
                        'p_value': p_val,
                        'label': lag_label,
                        'n_observations': len(corr_data)
                    }
                    
                except Exception as e:
                    continue
                    
            # Find strongest correlation across lags
            if lag_correlations:
                best_lag = max(lag_correlations.keys(), key=lambda x: abs(lag_correlations[x]['correlation']))
                best_corr = lag_correlations[best_lag]
                
                print(f"   🎯 Best correlation: {best_corr['label']}")
                print(f"      r = {best_corr['correlation']:+.3f} (p = {best_corr['p_value']:.4f})")
                
                # Display all significant lags
                significant_lags = {k: v for k, v in lag_correlations.items() if abs(v['correlation']) > 0.15 or v['p_value'] < 0.1}
                
                if significant_lags:
                    print(f"   📈 All significant lags:")
                    for lag, stats in sorted(significant_lags.items()):
                        print(f"      Lag {lag:+2d}: r={stats['correlation']:+.3f} (p={stats['p_value']:.3f}) - {stats['label']}")
                        
                lead_lag_results[fin_var] = {
                    'all_lags': lag_correlations,
                    'best_lag': best_lag,
                    'best_correlation': best_corr,
                    'significant_lags': significant_lags
                }
                
        # Store results
        self.lead_lag_results[result_key] = lead_lag_results
        
        return lead_lag_results
        
    def perform_granger_causality_tests(self, asset_name, subreddit_filter=None, max_lags=4):
        """📊 Test di causalità di Granger"""
        
        print(f"\n📊 === GRANGER CAUSALITY TESTS: {asset_name.upper()} ===")
        
        result_key = f"{asset_name}_{subreddit_filter if subreddit_filter else 'overall'}"
        
        if result_key not in self.correlation_results:
            print(f"❌ No correlation results found. Run analyze_asset_correlations first.")
            return None
            
        merged = self.correlation_results[result_key]['merged_data'].copy()
        
        # Select variables
        if subreddit_filter:
            sentiment_var = 'sentiment_mean'
        else:
            sentiment_var = 'overall_sentiment_mean'
            
        financial_vars = ['daily_return', 'return_abs']
        
        granger_results = {}
        
        for fin_var in financial_vars:
            if fin_var not in merged.columns:
                continue
                
            # Prepare data
            test_data = merged[[sentiment_var, fin_var]].dropna()
            
            if len(test_data) < 20:  # Need sufficient data
                continue
                
            print(f"\n🔍 Testing: {sentiment_var} → {fin_var}")
            
            try:
                # Test if sentiment Granger-causes financial variable
                gc_result = grangercausalitytests(
                    test_data[[fin_var, sentiment_var]], 
                    maxlag=max_lags, 
                    verbose=False
                )
                
                # Extract p-values for each lag
                lag_results = {}
                for lag in range(1, max_lags + 1):
                    if lag in gc_result:
                        # Get F-test p-value
                        f_test_p = gc_result[lag][0]['ssr_ftest'][1]
                        # Get Chi-square test p-value  
                        chi2_test_p = gc_result[lag][0]['ssr_chi2test'][1]
                        
                        lag_results[lag] = {
                            'f_test_p': f_test_p,
                            'chi2_test_p': chi2_test_p,
                            'significant_f': f_test_p < 0.05,
                            'significant_chi2': chi2_test_p < 0.05
                        }
                        
                        significance = ""
                        if f_test_p < 0.01:
                            significance = "**"
                        elif f_test_p < 0.05:
                            significance = "*"
                            
                        print(f"   Lag {lag}: F-test p={f_test_p:.4f}{significance}, χ²-test p={chi2_test_p:.4f}")
                        
                # Find most significant lag
                significant_lags = {k: v for k, v in lag_results.items() if v['significant_f']}
                
                if significant_lags:
                    best_lag = min(significant_lags.keys(), key=lambda x: lag_results[x]['f_test_p'])
                    print(f"   🎯 Best lag: {best_lag} days (p={lag_results[best_lag]['f_test_p']:.4f})")
                    print(f"   ✅ Evidence of Granger causality: sentiment → {fin_var}")
                else:
                    print(f"   ❌ No significant Granger causality found")
                    
                granger_results[fin_var] = {
                    'lag_results': lag_results,
                    'significant_lags': significant_lags,
                    'has_causality': len(significant_lags) > 0,
                    'best_lag': best_lag if significant_lags else None
                }
                
            except Exception as e:
                print(f"   ❌ Granger test failed: {e}")
                
        # Store results
        if result_key not in self.statistical_tests:
            self.statistical_tests[result_key] = {}
        self.statistical_tests[result_key]['granger'] = granger_results
        
        return granger_results
        
    def generate_comprehensive_report(self):
        """📋 Genera report completo delle correlazioni"""
        
        print(f"\n📋 === COMPREHENSIVE CORRELATION REPORT ===")
        
        if not self.correlation_results:
            print("❌ No correlation results to report")
            return
            
        print(f"\n🎯 EXECUTIVE SUMMARY:")
        print("=" * 50)
        
        total_assets = len(set(key.split('_')[0] for key in self.correlation_results.keys()))
        total_tests = len(self.correlation_results)
        total_significant = sum(result['significant_count'] for result in self.correlation_results.values())
        total_correlations = sum(result['total_correlations'] for result in self.correlation_results.values())
        
        print(f"📊 Assets analyzed: {total_assets}")
        print(f"🔍 Correlation tests performed: {total_tests}")
        print(f"✅ Significant correlations found: {total_significant}")
        print(f"📈 Total correlations analyzed: {total_correlations}")
        
        # Find strongest correlations across all assets
        all_strong_correlations = []
        
        for key, results in self.correlation_results.items():
            asset_name = key.split('_')[0]
            subreddit = '_'.join(key.split('_')[1:]) if '_' in key else 'overall'
            
            for corr_name, stats in results['correlations'].items():
                if abs(stats['pearson_corr']) > 0.2 or stats['pearson_p'] < 0.05:
                    all_strong_correlations.append({
                        'asset': asset_name,
                        'subreddit': subreddit,
                        'correlation_name': corr_name,
                        'correlation': stats['pearson_corr'],
                        'p_value': stats['pearson_p'],
                        'significance': stats['significance']
                    })
                    
        # Sort by absolute correlation
        all_strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print(f"\n🏆 TOP 10 STRONGEST CORRELATIONS ACROSS ALL ASSETS:")
        print("-" * 80)
        
        for i, corr in enumerate(all_strong_correlations[:10], 1):
            direction = "📈" if corr['correlation'] > 0 else "📉"
            strength = "STRONG" if abs(corr['correlation']) > 0.3 else "MODERATE"
            
            print(f"{i:2d}. {corr['asset'].upper()} ({corr['subreddit']}):")
            print(f"    {corr['correlation_name']}")
            print(f"    {direction} {strength} | r={corr['correlation']:+.3f} | p={corr['p_value']:.4f} | {corr['significance']}")
            
        # Asset-specific insights
        print(f"\n💡 KEY INSIGHTS BY ASSET:")
        print("=" * 50)
        
        for key, results in self.correlation_results.items():
            asset_name = key.split('_')[0]
            subreddit = '_'.join(key.split('_')[1:]) if '_' in key else 'overall'
            
            print(f"\n📊 {asset_name.upper()} (via r/{subreddit}):")
            
            top_3 = results['sorted_correlations'][:3]
            for i, (corr_name, stats) in enumerate(top_3, 1):
                corr_val = stats['pearson_corr']
                p_val = stats['pearson_p']
                
                print(f"   {i}. {corr_name}: r={corr_val:+.3f} (p={p_val:.3f})")
                
            # Data quality
            print(f"   📅 Data points: {results['data_points']} days")
            print(f"   ✅ Significant correlations: {results['significant_count']}")
            
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export correlation matrices for further analysis
        for key, results in self.correlation_results.items():
            if 'merged_data' in results:
                output_file = f"data/processed/correlation_analysis_{key}_{timestamp}.csv"
                results['merged_data'].to_csv(output_file, index=False)
                print(f"\n💾 Saved detailed data: {output_file}")
                
        print(f"\n🎉 CORRELATION ANALYSIS COMPLETED!")
        print(f"Ready for PHASE 2: Predictive Modeling with discovered correlations! 🚀")
        
        return {
            'summary': {
                'total_assets': total_assets,
                'total_tests': total_tests,
                'significant_correlations': total_significant,
                'total_correlations': total_correlations
            },
            'top_correlations': all_strong_correlations,
            'detailed_results': self.correlation_results,
            'statistical_tests': self.statistical_tests,
            'lead_lag_results': self.lead_lag_results
        }
        
    def run_complete_analysis(self):
        """🚀 Esegue analisi completa delle correlazioni"""
        
        print("🚀 === STARTING COMPREHENSIVE CORRELATION ANALYSIS ===\n")
        
        # Load data
        if not self.load_sentiment_data():
            print("❌ Failed to load sentiment data")
            return False
            
        if not self.load_financial_data():
            print("❌ Failed to load financial data")
            return False
            
        print("\n" + "="*60)
        print("🔍 STARTING CORRELATION ANALYSIS FOR ALL ASSETS")
        print("="*60)
        
        # Define asset-subreddit mappings for targeted analysis
        analysis_pairs = [
            # Crypto correlations (perfect alignment)
            ('bitcoin', 'CryptoCurrency'),
            ('ethereum', 'CryptoCurrency'),
            
            # Stock correlations  
            ('sp500', 'wallstreetbets'),
            ('sp500', 'investing'),
            ('sp500', 'stocks'),
            ('nasdaq', 'wallstreetbets'),
            ('nasdaq', 'investing'),
            ('nasdaq', 'stocks'),
            
            # Volatility analysis
            ('vix', None),  # Overall sentiment vs VIX
            
            # Dollar analysis
            ('dxy', None)   # Overall sentiment vs Dollar
        ]
        
        # Run correlation analysis for each pair
        for asset, subreddit in analysis_pairs:
            if asset in self.financial_data:
                print(f"\n{'='*60}")
                
                # Basic correlation analysis
                self.analyze_asset_correlations(asset, subreddit)
                
                # Lead-lag analysis
                self.analyze_lead_lag_relationships(asset, subreddit, max_lags=3)
                
                # Granger causality (for most promising pairs)
                if asset in ['bitcoin', 'ethereum', 'vix']:
                    self.perform_granger_causality_tests(asset, subreddit, max_lags=3)
                    
        # Generate comprehensive report
        final_report = self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_correlation_visualizations()
        
        print(f"\n✅ === CORRELATION ANALYSIS PHASE 1 COMPLETED ===")
        print(f"🎯 Ready to proceed to PHASE 2: Predictive Modeling")
        print(f"📊 Use discovered correlations to guide LSTM architecture")
        
        return final_report
        
    def create_correlation_visualizations(self):
        """📊 Crea visualizzazioni delle correlazioni"""
        
        print(f"\n📊 Creating correlation visualizations...")
        
        # Ensure results directory exists
        os.makedirs("results/correlation_analysis", exist_ok=True)
        
        try:
            # 1. Overall correlation heatmap
            self._create_correlation_heatmap()
            
            # 2. Time series plots for strongest correlations
            self._create_time_series_plots()
            
            # 3. Lead-lag analysis plots
            self._create_lead_lag_plots()
            
            print(f"✅ Visualizations saved to results/correlation_analysis/")
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            
    def _create_correlation_heatmap(self):
        """📈 Crea heatmap delle correlazioni principali"""
        
        # Collect all significant correlations
        correlation_matrix_data = {}
        
        for key, results in self.correlation_results.items():
            asset_name = key.split('_')[0]
            subreddit = '_'.join(key.split('_')[1:]) if '_' in key else 'overall'
            
            # Get top 3 correlations
            top_correlations = results['sorted_correlations'][:3]
            
            for corr_name, stats in top_correlations:
                if abs(stats['pearson_corr']) > 0.15:  # Only meaningful correlations
                    row_name = f"{asset_name}_{subreddit}"
                    col_name = corr_name.split('_vs_')[1] if '_vs_' in corr_name else corr_name
                    correlation_matrix_data[(row_name, col_name)] = stats['pearson_corr']
                    
        if correlation_matrix_data:
            # Convert to DataFrame for heatmap
            rows = list(set(item[0] for item in correlation_matrix_data.keys()))
            cols = list(set(item[1] for item in correlation_matrix_data.keys()))
            
            matrix = np.zeros((len(rows), len(cols)))
            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    if (row, col) in correlation_matrix_data:
                        matrix[i, j] = correlation_matrix_data[(row, col)]
                        
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                matrix, 
                xticklabels=cols, 
                yticklabels=rows,
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r', 
                center=0,
                cbar_kws={'label': 'Pearson Correlation Coefficient'}
            )
            plt.title('Sentiment-Financial Correlations Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Financial Metrics', fontweight='bold')
            plt.ylabel('Asset-Sentiment Pairs', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('results/correlation_analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_time_series_plots(self):
        """📈 Crea grafici time series per correlazioni più forti"""
        
        # Find strongest correlations for visualization
        strongest_correlations = []
        
        for key, results in self.correlation_results.items():
            if results['sorted_correlations']:
                top_corr = results['sorted_correlations'][0]
                if abs(top_corr[1]['pearson_corr']) > 0.2:  # Only strong correlations
                    strongest_correlations.append((key, top_corr))
                    
        # Create plots for top correlations
        for i, (key, (corr_name, stats)) in enumerate(strongest_correlations[:4]):  # Top 4
            asset_name = key.split('_')[0]
            subreddit = '_'.join(key.split('_')[1:]) if '_' in key else 'overall'
            
            # Get data
            merged_data = self.correlation_results[key]['merged_data']
            
            # Extract variable names
            sent_var = corr_name.split('_vs_')[0]
            fin_var = corr_name.split('_vs_')[1]
            
            if sent_var in merged_data.columns and fin_var in merged_data.columns:
                # Create subplot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Plot sentiment
                ax1.plot(merged_data['Date'], merged_data[sent_var], 'b-', alpha=0.7, linewidth=2)
                ax1.set_ylabel(f'Sentiment ({sent_var})', color='b', fontweight='bold')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True, alpha=0.3)
                
                # Plot financial metric
                ax2.plot(merged_data['Date'], merged_data[fin_var], 'r-', alpha=0.7, linewidth=2)
                ax2.set_ylabel(f'Financial ({fin_var})', color='r', fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.set_xlabel('Date', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Title and formatting
                title = f'{asset_name.upper()} ({subreddit}): {sent_var} vs {fin_var}\nCorrelation: r = {stats["pearson_corr"]:.3f} (p = {stats["pearson_p"]:.3f})'
                fig.suptitle(title, fontsize=12, fontweight='bold')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'results/correlation_analysis/timeseries_{asset_name}_{subreddit}_{i+1}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
    def _create_lead_lag_plots(self):
        """⏰ Crea grafici lead-lag analysis"""
        
        if not self.lead_lag_results:
            return
            
        for key, lead_lag_data in self.lead_lag_results.items():
            asset_name = key.split('_')[0]
            subreddit = '_'.join(key.split('_')[1:]) if '_' in key else 'overall'
            
            # Create lead-lag plot for each financial variable
            for fin_var, lag_results in lead_lag_data.items():
                if 'all_lags' not in lag_results:
                    continue
                    
                lags = list(lag_results['all_lags'].keys())
                correlations = [lag_results['all_lags'][lag]['correlation'] for lag in lags]
                p_values = [lag_results['all_lags'][lag]['p_value'] for lag in lags]
                
                # Create plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Correlation plot
                bars1 = ax1.bar(lags, correlations, alpha=0.7, 
                               color=['red' if c < 0 else 'blue' for c in correlations])
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.set_ylabel('Correlation Coefficient', fontweight='bold')
                ax1.set_title(f'{asset_name.upper()} ({subreddit}): Lead-Lag Analysis\nSentiment vs {fin_var}', 
                             fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # P-value plot
                bars2 = ax2.bar(lags, p_values, alpha=0.7, 
                               color=['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values])
                ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
                ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='p=0.10')
                ax2.set_ylabel('P-value', fontweight='bold')
                ax2.set_xlabel('Lag (days)\nNegative: Financial leads | Positive: Sentiment leads', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'results/correlation_analysis/leadlag_{asset_name}_{subreddit}_{fin_var}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()

def main():
    """🚀 Main execution function"""
    
    print("🚀 === PHASE 1: COMPREHENSIVE CORRELATION ANALYSIS ===")
    print("🎯 Objective: Validate dataset and discover sentiment-financial relationships")
    print("📊 This analysis will guide PHASE 2 (Predictive Modeling)\n")
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveCorrelationAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        if results:
            print(f"\n🎉 === PHASE 1 COMPLETED SUCCESSFULLY ===")
            print(f"✅ Correlation analysis finished")
            print(f"📊 Statistical tests performed")
            print(f"📈 Visualizations created")
            print(f"💾 Results saved for Phase 2")
            
            print(f"\n🚀 === READY FOR PHASE 2 ===")
            print(f"🔧 Use correlation insights to design LSTM architecture")
            print(f"🎯 Focus on assets with strongest correlations")
            print(f"⏰ Implement lead-lag relationships in predictive model")
            print(f"📊 Validate predictions against discovered patterns")
            
            return True
        else:
            print(f"\n❌ Phase 1 analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in correlation analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 === NEXT STEPS ===")
        print("1. 📋 Review correlation results in results/correlation_analysis/")
        print("2. 📊 Analyze visualizations to understand relationships")
        print("3. 🚀 Proceed to Phase 2: Predictive Modeling")
        print("4. 🏗️ Design LSTM architecture based on strongest correlations")
    else:
        print(f"\n🔧 === TROUBLESHOOTING ===")
        print("1. ✅ Check that FinBERT sentiment analysis was completed")
        print("2. ✅ Verify financial data was downloaded correctly")
        print("3. ✅ Ensure data alignment was successful")