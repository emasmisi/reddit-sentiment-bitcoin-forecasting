#!/usr/bin/env python3
"""
Setup script for massive Reddit data collection
Run this first to prepare your environment
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        'praw==7.7.1',
        'pandas==2.1.4',
        'python-dotenv==1.0.0',
        'schedule==1.2.0',
        'tqdm==4.66.1'
    ]
    
    print("📦 Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✅ Installed: {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {req}")

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data',
        'data/raw',
        'data/processed', 
        'data/financial',
        'logs',
        'models',
        'results',
        'notebooks'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")

def create_env_template():
    """Create .env template if it doesn't exist"""
    env_file = Path('.env')
    
    if not env_file.exists():
        env_content = """# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=script:ThesisMassiveCollection:v2.0 (by /u/your_username)

# Collection Configuration
TARGET_POSTS_PER_SUBREDDIT=2500
COLLECTION_PERIOD_MONTHS=6
BATCH_SIZE=100
RATE_LIMIT_DELAY=2.0

# Database Configuration
DATABASE_PATH=data/reddit_data.db

# Monitoring
LOG_LEVEL=INFO
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env template")
        print("⚠️  IMPORTANT: Update .env with your Reddit API credentials!")
    else:
        print("✅ .env file already exists")

def create_monitoring_script():
    """Create monitoring dashboard script"""
    monitoring_code = '''
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def create_collection_dashboard():
    """Create a dashboard to monitor collection progress"""
    
    # Connect to database
    conn = sqlite3.connect("data/reddit_data.db")
    
    # Get collection stats
    posts_df = pd.read_sql_query("""
        SELECT subreddit, 
               COUNT(*) as total_posts,
               MIN(created_date) as earliest_post,
               MAX(created_date) as latest_post,
               AVG(score) as avg_score,
               AVG(num_comments) as avg_comments
        FROM posts 
        GROUP BY subreddit
    """, conn)
    
    comments_df = pd.read_sql_query("""
        SELECT subreddit, COUNT(*) as total_comments
        FROM comments 
        GROUP BY subreddit
    """, conn)
    
    # Merge data
    stats = posts_df.merge(comments_df, on='subreddit', how='left')
    stats['total_comments'] = stats['total_comments'].fillna(0)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reddit Data Collection Dashboard', fontsize=16)
    
    # Posts by subreddit
    axes[0,0].bar(stats['subreddit'], stats['total_posts'])
    axes[0,0].set_title('Posts Collected by Subreddit')
    axes[0,0].set_ylabel('Number of Posts')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Comments by subreddit
    axes[0,1].bar(stats['subreddit'], stats['total_comments'])
    axes[0,1].set_title('Comments Collected by Subreddit')
    axes[0,1].set_ylabel('Number of Comments')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Average engagement
    axes[1,0].bar(stats['subreddit'], stats['avg_score'])
    axes[1,0].set_title('Average Post Score by Subreddit')
    axes[1,0].set_ylabel('Average Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Comments per post ratio
    stats['comments_per_post'] = stats['total_comments'] / stats['total_posts']
    axes[1,1].bar(stats['subreddit'], stats['comments_per_post'])
    axes[1,1].set_title('Comments per Post Ratio')
    axes[1,1].set_ylabel('Comments/Post')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/collection_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\\n📊 === COLLECTION SUMMARY ===")
    print(f"Total Posts: {stats['total_posts'].sum():,}")
    print(f"Total Comments: {stats['total_comments'].sum():,}")
    print(f"Total Data Points: {stats['total_posts'].sum() + stats['total_comments'].sum():,}")
    
    print("\\n📋 By Subreddit:")
    for _, row in stats.iterrows():
        print(f"  r/{row['subreddit']}: {int(row['total_posts']):,} posts, {int(row['total_comments']):,} comments")
    
    conn.close()
    return stats

if __name__ == "__main__":
    create_collection_dashboard()
'''
    
    with open('monitoring_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(monitoring_code)
    
    print("✅ Created monitoring dashboard script")

def check_reddit_credentials():
    """Check if Reddit credentials are configured"""
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT')
        
        if client_id and client_secret and user_agent:
            if 'your_' not in client_id and 'your_' not in client_secret:
                print("✅ Reddit credentials appear to be configured")
                return True
        
        print("⚠️  Reddit credentials not properly configured in .env")
        return False
        
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 === MASSIVE REDDIT DATA COLLECTION SETUP ===\\n")
    
    # Step 1: Install requirements
    install_requirements()
    print()
    
    # Step 2: Create directories
    create_directory_structure()
    print()
    
    # Step 3: Create .env template
    create_env_template()
    print()
    
    # Step 4: Create monitoring script
    create_monitoring_script()
    print()
    
    # Step 5: Check credentials
    credentials_ok = check_reddit_credentials()
    print()
    
    # Final instructions
    print("🎯 === SETUP COMPLETED ===")
    print()
    
    if not credentials_ok:
        print("❌ NEXT STEPS REQUIRED:")
        print("1. Update .env file with your Reddit API credentials")
        print("2. Get credentials from: https://www.reddit.com/prefs/apps")
        print("3. Create a 'script' type application")
        print()
    else:
        print("✅ ALL SETUP COMPLETE!")
        print()
    
    print("🚀 TO START MASSIVE COLLECTION:")
    print("   python massive_data_collector.py")
    print()
    print("📊 TO MONITOR PROGRESS:")
    print("   python monitoring_dashboard.py")
    print()
    print("🎯 TARGET: 10,000+ posts, 50,000+ comments")
    print("⏱️ ESTIMATED TIME: 6-12 hours")

if __name__ == "__main__":
    main()