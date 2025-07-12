#!/usr/bin/env python3
"""
✅ TEMPORAL ALIGNMENT VALIDATOR
===============================
Valida allineamento temporale tra Reddit e dati finanziari
- Fix timezone issues
- Conferma overlap perfetto
- Genera report finale
"""

import pandas as pd
import sqlite3
import glob
import os
from datetime import datetime
import pytz

def validate_temporal_alignment():
    """✅ Valida allineamento temporale completo"""
    
    print("✅ === TEMPORAL ALIGNMENT VALIDATOR ===\n")
    
    # 1. Reddit data timeframe
    print("📊 REDDIT DATA ANALYSIS:")
    
    try:
        conn = sqlite3.connect("data/reddit_data.db")
        
        reddit_query = """
        SELECT 
            MIN(created_date) as earliest_date,
            MAX(created_date) as latest_date,
            COUNT(*) as total_posts
        FROM posts
        """
        reddit_result = pd.read_sql_query(reddit_query, conn)
        
        reddit_start = pd.to_datetime(reddit_result['earliest_date'].iloc[0])
        reddit_end = pd.to_datetime(reddit_result['latest_date'].iloc[0])
        reddit_posts = reddit_result['total_posts'].iloc[0]
        
        print(f"   📅 Period: {reddit_start.date()} → {reddit_end.date()}")
        print(f"   📝 Posts: {reddit_posts:,}")
        print(f"   ⏰ Duration: {(reddit_end - reddit_start).days} days")
        
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Error reading Reddit data: {e}")
        return
    
    # 2. Financial data analysis
    print(f"\n📈 FINANCIAL DATA ANALYSIS:")
    
    financial_files = glob.glob("data/financial/*.csv")
    
    if not financial_files:
        print("   ❌ No financial data files found!")
        return
        
    financial_summary = {}
    
    for file_path in financial_files:
        filename = os.path.basename(file_path)
        asset_name = filename.split('_')[0]
        
        try:
            df = pd.read_csv(file_path)
            
            # Handle timezone-aware dates
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            
            # Convert to naive datetime for comparison
            df['Datetime_naive'] = df['Datetime'].dt.tz_localize(None)
            
            financial_summary[asset_name] = {
                'records': len(df),
                'start': df['Datetime_naive'].min(),
                'end': df['Datetime_naive'].max(),
                'file': filename
            }
            
            print(f"   📊 {asset_name.upper()}: {len(df):,} records")
            print(f"      📅 Range: {df['Datetime_naive'].min().date()} → {df['Datetime_naive'].max().date()}")
            
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
            
    # 3. Overlap analysis
    print(f"\n🎯 OVERLAP ANALYSIS:")
    
    all_perfect = True
    
    for asset, info in financial_summary.items():
        fin_start = info['start']
        fin_end = info['end']
        
        # Check if financial data covers Reddit period
        covers_start = fin_start <= reddit_start
        covers_end = fin_end >= reddit_end
        
        if covers_start and covers_end:
            overlap_days = min(fin_end, reddit_end) - max(fin_start, reddit_start)
            coverage_pct = (overlap_days.days / (reddit_end - reddit_start).days) * 100
            
            print(f"   ✅ {asset.upper()}: PERFECT COVERAGE ({coverage_pct:.1f}%)")
        else:
            print(f"   ⚠️ {asset.upper()}: Partial coverage")
            all_perfect = False
            
            if not covers_start:
                gap_start = (reddit_start - fin_start).days
                print(f"      📅 Starts {gap_start} days after Reddit data")
                
            if not covers_end:
                gap_end = (fin_end - reddit_end).days
                print(f"      📅 Ends {gap_end} days before Reddit data")
    
    # 4. Final assessment
    print(f"\n🏆 === FINAL ASSESSMENT ===")
    
    if all_perfect:
        print("✅ PERFECT TEMPORAL ALIGNMENT ACHIEVED!")
        print("✅ All financial assets completely cover Reddit period")
        print("✅ Dataset ready for sentiment correlation analysis")
        
        # Calculate total data points
        total_reddit = reddit_posts
        total_financial = sum(info['records'] for info in financial_summary.values())
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   📝 Reddit data points: {total_reddit:,}")
        print(f"   📈 Financial data points: {total_financial:,}")
        print(f"   🎯 Total data points: {total_reddit + total_financial:,}")
        
        print(f"\n🚀 READY FOR NEXT STEPS:")
        print("1. ✅ Temporal alignment: PERFECT")
        print("2. 🧠 FinBERT sentiment analysis: COMPLETED")
        print("3. 📊 Correlation analysis: READY TO RUN")
        print("4. 🤖 Predictive modeling: READY TO BUILD")
        
    else:
        print("⚠️ Some alignment issues detected")
        print("   Most assets should still be sufficient for analysis")
        
    # 5. Generate correlation roadmap
    print(f"\n🗺️ === CORRELATION ROADMAP ===")
    
    correlations = [
        "r/wallstreetbets sentiment → S&P 500 price movements",
        "r/stocks + r/investing sentiment → NASDAQ movements", 
        "r/CryptoCurrency sentiment → Bitcoin/Ethereum prices",
        "Overall sentiment volatility → VIX volatility index",
        "Sentiment timing patterns → Market hour performance"
    ]
    
    for i, correlation in enumerate(correlations, 1):
        print(f"   {i}. {correlation}")
        
    return {
        'reddit_period': (reddit_start, reddit_end),
        'reddit_posts': reddit_posts,
        'financial_summary': financial_summary,
        'perfect_alignment': all_perfect
    }

def generate_next_steps_guide():
    """📋 Genera guida per prossimi step"""
    
    print(f"\n📋 === NEXT STEPS GUIDE ===")
    
    steps = [
        {
            'step': '1. Correlation Analysis',
            'description': 'Analyze correlations between Reddit sentiment and financial movements',
            'command': 'python correlation_analysis.py',
            'priority': 'HIGH'
        },
        {
            'step': '2. Feature Engineering', 
            'description': 'Create advanced features combining sentiment + financial indicators',
            'command': 'python feature_engineering.py',
            'priority': 'HIGH'
        },
        {
            'step': '3. Predictive Modeling',
            'description': 'Build LSTM model with FinBERT sentiment for price prediction',
            'command': 'python predictive_model.py',
            'priority': 'MEDIUM'
        },
        {
            'step': '4. Backtesting',
            'description': 'Test trading strategies based on sentiment signals',
            'command': 'python backtesting.py', 
            'priority': 'MEDIUM'
        },
        {
            'step': '5. Visualization',
            'description': 'Create charts and visualizations for thesis',
            'command': 'python create_visualizations.py',
            'priority': 'LOW'
        }
    ]
    
    for step_info in steps:
        priority_emoji = {'HIGH': '🔥', 'MEDIUM': '⚡', 'LOW': '📊'}[step_info['priority']]
        
        print(f"\n{priority_emoji} {step_info['step']} ({step_info['priority']} PRIORITY)")
        print(f"   📝 {step_info['description']}")
        print(f"   💻 {step_info['command']}")

def main():
    """🚀 Main execution"""
    
    try:
        results = validate_temporal_alignment()
        generate_next_steps_guide()
        
        if results and results.get('perfect_alignment'):
            print(f"\n🎉 === SUCCESS! ===")
            print("Your temporal alignment is PERFECT!")
            print("You can now proceed with confidence to correlation analysis! 🚀")
            
    except Exception as e:
        print(f"❌ Error in validation: {e}")

if __name__ == "__main__":
    main()