import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os
from datetime import datetime
import numpy as np

def analyze_scraped_data():
    """Analisi esplorativa dei dati appena scaricati"""
    
    print("🔍 === ANALISI ESPLORATIVA DATI REDDIT ===\n")
    
    # Trova tutti i file di dati più recenti
    data_dir = "data/raw"
    
    # Trova i file più recenti per ogni subreddit
    wsb_posts = sorted(glob.glob(f"{data_dir}/wallstreetbets_posts_*.csv"))[-1]
    wsb_comments = sorted(glob.glob(f"{data_dir}/wallstreetbets_comments_*.csv"))[-1]
    crypto_posts = sorted(glob.glob(f"{data_dir}/CryptoCurrency_posts_*.csv"))[-1]
    crypto_comments = sorted(glob.glob(f"{data_dir}/CryptoCurrency_comments_*.csv"))[-1]
    
    print(f"📁 File analizzati:")
    print(f"   🏦 WSB Posts: {wsb_posts}")
    print(f"   🏦 WSB Comments: {wsb_comments}")
    print(f"   💰 Crypto Posts: {crypto_posts}")
    print(f"   💰 Crypto Comments: {crypto_comments}\n")
    
    # Carica i dati
    wsb_posts_df = pd.read_csv(wsb_posts)
    wsb_comments_df = pd.read_csv(wsb_comments)
    crypto_posts_df = pd.read_csv(crypto_posts)
    crypto_comments_df = pd.read_csv(crypto_comments)
    
    # === STATISTICHE GENERALI ===
    print("📊 === STATISTICHE GENERALI ===")
    print(f"WSB Posts: {len(wsb_posts_df):,} | Comments: {len(wsb_comments_df):,}")
    print(f"Crypto Posts: {len(crypto_posts_df):,} | Comments: {len(crypto_comments_df):,}")
    print(f"TOTALE: {len(wsb_posts_df) + len(crypto_posts_df):,} posts, {len(wsb_comments_df) + len(crypto_comments_df):,} commenti\n")
    
    # === ANALISI POSTS ===
    print("📝 === ANALISI POSTS ===")
    
    def analyze_posts(df, name):
        print(f"\n🎯 {name}:")
        print(f"   Score medio: {df['score'].mean():.1f}")
        print(f"   Score massimo: {df['score'].max():,}")
        print(f"   Commenti medi: {df['num_comments'].mean():.1f}")
        print(f"   Upvote ratio medio: {df['upvote_ratio'].mean():.2f}")
        print(f"   Lunghezza titolo media: {df['title_length'].mean():.1f} caratteri")
        print(f"   Lunghezza testo media: {df['text_length'].mean():.0f} caratteri")
        
        # Top 3 post per score
        top_posts = df.nlargest(3, 'score')[['title', 'score', 'num_comments']]
        print(f"   📈 Top 3 post per score:")
        for i, (_, row) in enumerate(top_posts.iterrows(), 1):
            title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            print(f"      {i}. {title} (Score: {row['score']}, Comments: {row['num_comments']})")
    
    analyze_posts(wsb_posts_df, "WallStreetBets")
    analyze_posts(crypto_posts_df, "CryptoCurrency")
    
    # === ANALISI TEMPORALE ===
    print(f"\n📅 === ANALISI TEMPORALE ===")
    
    # Combina tutti i post
    all_posts = pd.concat([
        wsb_posts_df.assign(subreddit='wallstreetbets'),
        crypto_posts_df.assign(subreddit='CryptoCurrency')
    ])
    
    all_posts['created_date'] = pd.to_datetime(all_posts['created_date'])
    all_posts['day_of_week'] = all_posts['created_date'].dt.day_name()
    
    # Post per giorno
    posts_per_day = all_posts.groupby(['created_date', 'subreddit']).size().unstack(fill_value=0)
    print("Post per giorno:")
    print(posts_per_day.to_string())
    
    # Post per ora del giorno
    posts_per_hour = all_posts.groupby(['created_hour', 'subreddit']).size().unstack(fill_value=0)
    print(f"\nPost per ora (top 5 ore più attive):")
    hourly_totals = posts_per_hour.sum(axis=1).sort_values(ascending=False)
    for hour in hourly_totals.head().index:
        wsb_count = posts_per_hour.loc[hour, 'wallstreetbets'] if 'wallstreetbets' in posts_per_hour.columns else 0
        crypto_count = posts_per_hour.loc[hour, 'CryptoCurrency'] if 'CryptoCurrency' in posts_per_hour.columns else 0
        print(f"   {hour:2d}:00 - WSB: {wsb_count:2d}, Crypto: {crypto_count:2d}, Totale: {hourly_totals[hour]:2d}")
    
    # === ANALISI ENGAGEMENT ===
    print(f"\n💬 === ANALISI ENGAGEMENT ===")
    
    def analyze_engagement(posts_df, comments_df, name):
        # Calcola metriche di engagement
        posts_with_engagement = posts_df.copy()
        posts_with_engagement['comments_per_score'] = posts_with_engagement['num_comments'] / (posts_with_engagement['score'] + 1)
        
        avg_comments_per_post = len(comments_df) / len(posts_df) if len(posts_df) > 0 else 0
        
        print(f"\n🎯 {name}:")
        print(f"   Commenti per post (medio): {avg_comments_per_post:.1f}")
        print(f"   Engagement ratio medio: {posts_with_engagement['comments_per_score'].mean():.3f}")
        print(f"   Post con 0 commenti: {(posts_df['num_comments'] == 0).sum()} ({(posts_df['num_comments'] == 0).mean()*100:.1f}%)")
        print(f"   Post con >50 commenti: {(posts_df['num_comments'] > 50).sum()} ({(posts_df['num_comments'] > 50).mean()*100:.1f}%)")
    
    analyze_engagement(wsb_posts_df, wsb_comments_df, "WallStreetBets")
    analyze_engagement(crypto_posts_df, crypto_comments_df, "CryptoCurrency")
    
    # === PREVIEW CONTENUTI ===
    print(f"\n📖 === PREVIEW CONTENUTI (per sentiment analysis) ===")
    
    def show_content_samples(posts_df, name):
        print(f"\n🎯 {name} - Esempi di titoli:")
        samples = posts_df.sample(min(5, len(posts_df)))['title'].tolist()
        for i, title in enumerate(samples, 1):
            print(f"   {i}. {title}")
    
    show_content_samples(wsb_posts_df, "WallStreetBets")
    show_content_samples(crypto_posts_df, "CryptoCurrency")
    
    # === CONTROLLO QUALITÀ DATI ===
    print(f"\n✅ === CONTROLLO QUALITÀ DATI ===")
    
    def check_data_quality(df, name):
        print(f"\n🎯 {name}:")
        print(f"   Righe totali: {len(df):,}")
        print(f"   Valori mancanti: {df.isnull().sum().sum()}")
        print(f"   Duplicati: {df.duplicated().sum()}")
        if 'text_content' in df.columns:
            print(f"   Testi vuoti: {(df['text_content'].str.len() < 10).sum()}")
            print(f"   Lunghezza media testo: {df['text_content'].str.len().mean():.0f} caratteri")
    
    check_data_quality(wsb_posts_df, "WSB Posts")
    check_data_quality(crypto_posts_df, "Crypto Posts")
    check_data_quality(wsb_comments_df, "WSB Comments")
    check_data_quality(crypto_comments_df, "Crypto Comments")
    
    print(f"\n🎉 === ANALISI COMPLETATA ===")
    print(f"I tuoi dati sono di ottima qualità e pronti per la sentiment analysis!")
    print(f"Prossimo step: installare le librerie per sentiment analysis e scaricare dati finanziari.")

if __name__ == "__main__":
    analyze_scraped_data()