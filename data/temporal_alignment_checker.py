#!/usr/bin/env python3
"""
📅 TEMPORAL ALIGNMENT CHECKER
=============================
Verifica allineamento temporale tra dati Reddit e dati finanziari
- Controlla range date Reddit
- Controlla range date dati finanziari  
- Identifica gap temporali
- Suggerisce correzioni
"""

import pandas as pd
import sqlite3
import glob
import os
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional

class TemporalAlignmentChecker:
    """📅 Checker per allineamento temporale dati"""
    
    def __init__(self):
        print("📅 === TEMPORAL ALIGNMENT CHECKER ===")
        
    def check_reddit_data_timeframe(self, db_path: str = "data/reddit_data.db") -> Dict:
        """📊 Controlla timeframe dati Reddit"""
        
        print("🔍 Controllando timeframe dati Reddit...")
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Posts timeframe
            posts_query = """
            SELECT 
                MIN(created_date) as earliest_post,
                MAX(created_date) as latest_post,
                COUNT(*) as total_posts
            FROM posts
            """
            posts_result = pd.read_sql_query(posts_query, conn)
            
            # Comments timeframe  
            comments_query = """
            SELECT 
                MIN(created_date) as earliest_comment,
                MAX(created_date) as latest_comment,
                COUNT(*) as total_comments
            FROM comments
            """
            comments_result = pd.read_sql_query(comments_query, conn)
            
            # Per subreddit
            subreddit_query = """
            SELECT 
                subreddit,
                MIN(created_date) as earliest,
                MAX(created_date) as latest,
                COUNT(*) as count
            FROM posts
            GROUP BY subreddit
            """
            subreddit_result = pd.read_sql_query(subreddit_query, conn)
            
            conn.close()
            
            reddit_info = {
                'posts_earliest': posts_result['earliest_post'].iloc[0],
                'posts_latest': posts_result['latest_post'].iloc[0],
                'posts_total': posts_result['total_posts'].iloc[0],
                'comments_earliest': comments_result['earliest_comment'].iloc[0],
                'comments_latest': comments_result['latest_comment'].iloc[0],
                'comments_total': comments_result['total_comments'].iloc[0],
                'subreddit_breakdown': subreddit_result
            }
            
            print(f"✅ Reddit data timeframe:")
            print(f"   📝 Posts: {reddit_info['posts_earliest']} → {reddit_info['posts_latest']} ({reddit_info['posts_total']:,} posts)")
            print(f"   💬 Comments: {reddit_info['comments_earliest']} → {reddit_info['comments_latest']} ({reddit_info['comments_total']:,} comments)")
            
            print(f"\n📊 Per subreddit:")
            for _, row in subreddit_result.iterrows():
                print(f"   r/{row['subreddit']}: {row['earliest']} → {row['latest']} ({row['count']:,} posts)")
                
            return reddit_info
            
        except Exception as e:
            print(f"❌ Errore nel controllo Reddit: {e}")
            return {}
            
    def check_financial_data_timeframe(self, data_dir: str = "data/financial") -> Dict:
        """📈 Controlla timeframe dati finanziari esistenti"""
        
        print(f"\n🔍 Controllando timeframe dati finanziari in {data_dir}...")
        
        financial_info = {}
        
        if not os.path.exists(data_dir):
            print(f"❌ Directory {data_dir} non trovata!")
            return {}
            
        # Cerca file CSV esistenti
        csv_files = glob.glob(f"{data_dir}/*.csv")
        
        if not csv_files:
            print(f"❌ Nessun file CSV trovato in {data_dir}")
            return {}
            
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            asset_name = filename.split('_')[0] if '_' in filename else filename.replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path)
                
                # Trova colonna data
                date_col = None
                for col in ['Date', 'Datetime', 'date', 'datetime', 'timestamp']:
                    if col in df.columns:
                        date_col = col
                        break
                        
                if not date_col:
                    print(f"⚠️ {filename}: Nessuna colonna data trovata")
                    continue
                    
                df[date_col] = pd.to_datetime(df[date_col])
                
                earliest = df[date_col].min()
                latest = df[date_col].max()
                total_records = len(df)
                
                financial_info[asset_name] = {
                    'file': filename,
                    'earliest': earliest.strftime('%Y-%m-%d'),
                    'latest': latest.strftime('%Y-%m-%d'), 
                    'total_records': total_records,
                    'date_column': date_col
                }
                
                print(f"   📈 {asset_name}: {earliest.strftime('%Y-%m-%d')} → {latest.strftime('%Y-%m-%d')} ({total_records:,} records)")
                
            except Exception as e:
                print(f"❌ Errore leggendo {filename}: {e}")
                
        return financial_info
        
    def download_fresh_financial_data(self, reddit_start: str, reddit_end: str) -> Dict:
        """💰 Scarica dati finanziari allineati con periodo Reddit"""
        
        print(f"\n💰 Scaricando dati finanziari per periodo {reddit_start} → {reddit_end}...")
        
        # Aggiungi buffer di qualche giorno prima/dopo
        start_date = pd.to_datetime(reddit_start) - timedelta(days=7)
        end_date = pd.to_datetime(reddit_end) + timedelta(days=1)
        
        tickers = {
            'sp500': '^GSPC',    # S&P 500
            'nasdaq': '^IXIC',   # NASDAQ
            'bitcoin': 'BTC-USD', # Bitcoin
            'ethereum': 'ETH-USD', # Ethereum
            'vix': '^VIX'        # Volatility Index
        }
        
        fresh_data = {}
        
        for asset_name, ticker in tickers.items():
            try:
                print(f"   📊 Scaricando {asset_name} ({ticker})...")
                
                # Download con yfinance
                data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1h'  # Dati orari per correlazione precisa
                )
                
                if len(data) == 0:
                    print(f"   ⚠️ {asset_name}: Nessun dato disponibile")
                    continue
                    
                # Reset index per avere Datetime come colonna
                data.reset_index(inplace=True)
                
                # Aggiungi features
                data['price_change'] = data['Close'].pct_change()
                data['volatility'] = data['price_change'].rolling(24).std()  # 24h rolling volatility
                data['volume_ma'] = data['Volume'].rolling(24).mean() if 'Volume' in data.columns else 0
                
                fresh_data[asset_name] = {
                    'data': data,
                    'earliest': data['Datetime'].min().strftime('%Y-%m-%d %H:%M'),
                    'latest': data['Datetime'].max().strftime('%Y-%m-%d %H:%M'),
                    'total_records': len(data)
                }
                
                print(f"   ✅ {asset_name}: {len(data):,} record scaricati")
                
                # Salva file aggiornato
                os.makedirs("data/financial", exist_ok=True)
                output_file = f"data/financial/{asset_name}_hourly_{datetime.now().strftime('%Y%m%d')}.csv"
                data.to_csv(output_file, index=False)
                print(f"   💾 Salvato: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Errore con {asset_name}: {e}")
                
        return fresh_data
        
    def analyze_temporal_overlap(self, reddit_info: Dict, financial_info: Dict) -> Dict:
        """🔍 Analizza overlap temporale"""
        
        print(f"\n🔍 === ANALISI OVERLAP TEMPORALE ===")
        
        if not reddit_info or not financial_info:
            print("❌ Dati insufficienti per analisi overlap")
            return {}
            
        reddit_start = pd.to_datetime(reddit_info['posts_earliest'])
        reddit_end = pd.to_datetime(reddit_info['posts_latest'])
        
        overlap_analysis = {
            'reddit_period': {
                'start': reddit_start.strftime('%Y-%m-%d'),
                'end': reddit_end.strftime('%Y-%m-%d'),
                'duration_days': (reddit_end - reddit_start).days
            },
            'financial_assets': {},
            'overlap_status': {}
        }
        
        for asset, info in financial_info.items():
            fin_start = pd.to_datetime(info['earliest'])
            fin_end = pd.to_datetime(info['latest'])
            
            # Calcola overlap
            overlap_start = max(reddit_start, fin_start)
            overlap_end = min(reddit_end, fin_end)
            
            has_overlap = overlap_start <= overlap_end
            overlap_days = (overlap_end - overlap_start).days if has_overlap else 0
            
            overlap_analysis['financial_assets'][asset] = {
                'start': fin_start.strftime('%Y-%m-%d'),
                'end': fin_end.strftime('%Y-%m-%d'),
                'duration_days': (fin_end - fin_start).days
            }
            
            overlap_analysis['overlap_status'][asset] = {
                'has_overlap': has_overlap,
                'overlap_days': overlap_days,
                'overlap_percentage': (overlap_days / (reddit_end - reddit_start).days * 100) if has_overlap else 0,
                'gap_before': (reddit_start - fin_end).days if reddit_start > fin_end else 0,
                'gap_after': (fin_start - reddit_end).days if fin_start > reddit_end else 0
            }
            
            # Stampa risultati
            if has_overlap:
                print(f"✅ {asset}: OVERLAP {overlap_days} giorni ({overlap_analysis['overlap_status'][asset]['overlap_percentage']:.1f}%)")
            else:
                gap_before = overlap_analysis['overlap_status'][asset]['gap_before']
                gap_after = overlap_analysis['overlap_status'][asset]['gap_after']
                
                if gap_before > 0:
                    print(f"❌ {asset}: GAP {gap_before} giorni PRIMA dei dati Reddit")
                elif gap_after > 0:
                    print(f"❌ {asset}: GAP {gap_after} giorni DOPO i dati Reddit")
                else:
                    print(f"❌ {asset}: NESSUN OVERLAP")
                    
        return overlap_analysis
        
    def generate_alignment_strategy(self, overlap_analysis: Dict) -> List[str]:
        """🎯 Genera strategia di allineamento"""
        
        print(f"\n🎯 === STRATEGIA DI ALLINEAMENTO ===")
        
        strategies = []
        
        reddit_start = overlap_analysis['reddit_period']['start']
        reddit_end = overlap_analysis['reddit_period']['end']
        
        # Controlla overlap status
        good_overlaps = []
        need_update = []
        
        for asset, status in overlap_analysis['overlap_status'].items():
            if status['has_overlap'] and status['overlap_percentage'] > 80:
                good_overlaps.append(asset)
            else:
                need_update.append(asset)
                
        if good_overlaps:
            print(f"✅ Asset con buon overlap: {', '.join(good_overlaps)}")
            strategies.append(f"Usa dati esistenti per: {', '.join(good_overlaps)}")
            
        if need_update:
            print(f"⚠️ Asset da aggiornare: {', '.join(need_update)}")
            strategies.append(f"Scarica dati aggiornati per periodo {reddit_start} → {reddit_end}")
            
        # Raccomandazioni specifiche
        print(f"\n📋 RACCOMANDAZIONI:")
        
        if len(need_update) > len(good_overlaps):
            print("1. 🔄 DOWNLOAD COMPLETO: Scarica tutti i dati finanziari per il periodo Reddit")
            strategies.append("download_fresh_data")
            
        print("2. ⏰ ALLINEAMENTO ORARIO: Usa dati orari per correlazione precisa")
        strategies.append("hourly_alignment")
        
        print("3. 🎯 BUFFER TEMPORALE: Includi 1-2 giorni extra prima/dopo")
        strategies.append("temporal_buffer")
        
        print("4. 📊 VALIDAZIONE: Verifica qualità dati dopo download")
        strategies.append("data_validation")
        
        return strategies
        
    def run_complete_check(self):
        """🚀 Esegue check completo"""
        
        print("🚀 === CONTROLLO COMPLETO ALLINEAMENTO TEMPORALE ===\n")
        
        # 1. Check Reddit data
        reddit_info = self.check_reddit_data_timeframe()
        
        # 2. Check existing financial data
        financial_info = self.check_financial_data_timeframe()
        
        # 3. Analyze overlap
        overlap_analysis = self.analyze_temporal_overlap(reddit_info, financial_info)
        
        # 4. Generate strategy
        strategies = self.generate_alignment_strategy(overlap_analysis)
        
        # 5. Offer to download fresh data
        if reddit_info and ('download_fresh_data' in strategies or not financial_info):
            
            response = input(f"\n💰 Vuoi scaricare dati finanziari aggiornati per il periodo Reddit? (y/n): ").strip().lower()
            
            if response == 'y':
                fresh_data = self.download_fresh_financial_data(
                    reddit_info['posts_earliest'], 
                    reddit_info['posts_latest']
                )
                
                if fresh_data:
                    print(f"\n✅ === DOWNLOAD COMPLETATO ===")
                    print(f"Dati finanziari aggiornati e salvati in data/financial/")
                    print(f"Ora i tuoi dati Reddit e finanziari sono ALLINEATI! 🎯")
                    
        return {
            'reddit_info': reddit_info,
            'financial_info': financial_info,
            'overlap_analysis': overlap_analysis,
            'strategies': strategies
        }

def main():
    """🚀 Main execution"""
    
    checker = TemporalAlignmentChecker()
    results = checker.run_complete_check()
    
    print(f"\n🎉 === CHECK COMPLETATO ===")
    print("Usa i risultati per assicurarti che Reddit e dati finanziari siano allineati!")

if __name__ == "__main__":
    main()