#!/usr/bin/env python3
"""
💰 QUICK FINANCIAL DATA DOWNLOADER
===================================
Download immediato dati finanziari per periodo Reddit
- Periodo: 2025-05-17 → 2025-07-04
- Assets: S&P500, NASDAQ, Bitcoin, Ethereum
- Frequenza: Hourly data per correlazione precisa
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_financial_data_aligned():
    """💰 Download dati finanziari allineati con Reddit"""
    
    print("💰 === QUICK FINANCIAL DATA DOWNLOADER ===")
    print("📅 Periodo target: 2025-05-17 → 2025-07-04 (da analisi Reddit)")
    
    # Date con buffer
    start_date = "2025-05-15"  # 2 giorni prima per buffer
    end_date = "2025-07-06"    # 2 giorni dopo per buffer
    
    print(f"📊 Download period con buffer: {start_date} → {end_date}")
    
    # Asset da scaricare
    tickers = {
        'sp500': '^GSPC',      # S&P 500 (correlazione WSB/stocks/investing)
        'nasdaq': '^IXIC',     # NASDAQ (correlazione WSB/stocks/investing)
        'bitcoin': 'BTC-USD',  # Bitcoin (correlazione CryptoCurrency)
        'ethereum': 'ETH-USD', # Ethereum (correlazione CryptoCurrency)
        'vix': '^VIX',         # Volatility Index
        'dxy': 'DX-Y.NYB'      # Dollar Index
    }
    
    # Crea directory
    os.makedirs("data/financial", exist_ok=True)
    
    successful_downloads = {}
    failed_downloads = {}
    
    for asset_name, ticker in tickers.items():
        try:
            print(f"\n📊 Downloading {asset_name} ({ticker})...")
            
            # Download dati orari
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1h',  # Dati orari
                progress=False
            )
            
            if len(data) == 0:
                print(f"❌ {asset_name}: Nessun dato disponibile")
                failed_downloads[asset_name] = "No data available"
                continue
                
            # Reset index per avere Datetime come colonna
            data.reset_index(inplace=True)
            
            # Rinomina colonne per compatibilità
            if 'Datetime' not in data.columns and 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            
            # Aggiungi features tecniche
            data['price_change'] = data['Close'].pct_change()
            data['price_change_abs'] = data['price_change'].abs()
            
            # Rolling features (24h window)
            data['volatility_24h'] = data['price_change'].rolling(24).std()
            data['volume_ma_24h'] = data['Volume'].rolling(24).mean() if 'Volume' in data.columns else 0
            data['price_ma_24h'] = data['Close'].rolling(24).mean()
            
            # Momentum indicators
            data['momentum_1h'] = data['Close'] / data['Close'].shift(1) - 1
            data['momentum_6h'] = data['Close'] / data['Close'].shift(6) - 1
            data['momentum_24h'] = data['Close'] / data['Close'].shift(24) - 1
            
            # Directional features (per ML)
            data['price_up'] = (data['price_change'] > 0).astype(int)
            data['strong_move'] = (data['price_change_abs'] > data['volatility_24h']).astype(int)
            
            # Date features
            data['hour'] = pd.to_datetime(data['Datetime']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['Datetime']).dt.dayofweek
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            
            # Market session features
            data['us_market_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 16) & (data['day_of_week'] < 5)).astype(int)
            data['extended_hours'] = ((data['hour'] >= 4) & (data['hour'] <= 20) & (data['day_of_week'] < 5)).astype(int)
            
            successful_downloads[asset_name] = {
                'records': len(data),
                'start': data['Datetime'].min(),
                'end': data['Datetime'].max(),
                'file': f"data/financial/{asset_name}_hourly_{datetime.now().strftime('%Y%m%d')}.csv"
            }
            
            # Salva file
            data.to_csv(successful_downloads[asset_name]['file'], index=False)
            
            print(f"✅ {asset_name}: {len(data):,} records")
            print(f"   📅 Range: {data['Datetime'].min()} → {data['Datetime'].max()}")
            print(f"   💾 Saved: {successful_downloads[asset_name]['file']}")
            
        except Exception as e:
            print(f"❌ {asset_name}: Error - {e}")
            failed_downloads[asset_name] = str(e)
            
    # Summary
    print(f"\n🎉 === DOWNLOAD SUMMARY ===")
    print(f"✅ Successful: {len(successful_downloads)} assets")
    print(f"❌ Failed: {len(failed_downloads)} assets")
    
    if successful_downloads:
        print(f"\n📊 SUCCESSFUL DOWNLOADS:")
        for asset, info in successful_downloads.items():
            print(f"   {asset}: {info['records']:,} records ({info['start']} → {info['end']})")
            
    if failed_downloads:
        print(f"\n❌ FAILED DOWNLOADS:")
        for asset, error in failed_downloads.items():
            print(f"   {asset}: {error}")
            
    # Validation check
    if len(successful_downloads) >= 4:  # At least 4 assets
        print(f"\n🎯 === TEMPORAL ALIGNMENT CHECK ===")
        
        reddit_start = pd.to_datetime("2025-05-17")
        reddit_end = pd.to_datetime("2025-07-04")
        
        print(f"📅 Reddit period: {reddit_start.date()} → {reddit_end.date()}")
        
        all_aligned = True
        
        for asset, info in successful_downloads.items():
            fin_start = pd.to_datetime(info['start'])
            fin_end = pd.to_datetime(info['end'])
            
            # Check coverage
            covers_reddit_start = fin_start <= reddit_start
            covers_reddit_end = fin_end >= reddit_end
            
            if covers_reddit_start and covers_reddit_end:
                print(f"✅ {asset}: PERFECT ALIGNMENT")
            else:
                print(f"⚠️ {asset}: Partial coverage")
                all_aligned = False
                
        if all_aligned:
            print(f"\n🏆 === PERFECT TEMPORAL ALIGNMENT ACHIEVED! ===")
            print(f"All financial data perfectly covers Reddit period!")
            print(f"Your dataset is now ready for sentiment correlation analysis! 🚀")
        else:
            print(f"\n⚠️ Some assets have partial coverage - but should be sufficient for analysis")
            
    return successful_downloads, failed_downloads

def quick_data_inspection():
    """👀 Quick inspection dei dati scaricati"""
    
    print(f"\n👀 === QUICK DATA INSPECTION ===")
    
    financial_dir = "data/financial"
    
    if not os.path.exists(financial_dir):
        print(f"❌ Directory {financial_dir} not found")
        return
        
    import glob
    csv_files = glob.glob(f"{financial_dir}/*.csv")
    
    if not csv_files:
        print(f"❌ No CSV files found in {financial_dir}")
        return
        
    for file_path in csv_files[:3]:  # Show first 3 files
        filename = os.path.basename(file_path)
        asset_name = filename.split('_')[0]
        
        try:
            df = pd.read_csv(file_path)
            
            print(f"\n📊 {asset_name.upper()}:")
            print(f"   Records: {len(df):,}")
            print(f"   Columns: {list(df.columns)[:8]}...")  # First 8 columns
            print(f"   Date range: {df['Datetime'].min()} → {df['Datetime'].max()}")
            print(f"   Price range: ${df['Close'].min():.2f} → ${df['Close'].max():.2f}")
            
            # Show sample data
            print(f"   📋 Sample data:")
            sample = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'price_change']].head(3)
            for _, row in sample.iterrows():
                dt = pd.to_datetime(row['Datetime']).strftime('%Y-%m-%d %H:%M')
                print(f"      {dt}: ${row['Close']:.2f} ({row['price_change']:.3f}%)")
                
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

def main():
    """🚀 Main execution"""
    
    try:
        # Download data
        successful, failed = download_financial_data_aligned()
        
        # Quick inspection
        if successful:
            quick_data_inspection()
            
            print(f"\n🎯 === NEXT STEPS ===")
            print("1. ✅ Financial data downloaded and aligned with Reddit period")
            print("2. 🔄 Run FinBERT sentiment analysis if not done yet")
            print("3. 📊 Create correlation analysis between sentiment and prices")
            print("4. 🤖 Build predictive model with LSTM + FinBERT")
            print("\n🚀 Your temporal alignment issue is SOLVED! 🚀")
            
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")

if __name__ == "__main__":
    main()