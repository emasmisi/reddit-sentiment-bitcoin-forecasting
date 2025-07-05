
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
    print("\n📊 === COLLECTION SUMMARY ===")
    print(f"Total Posts: {stats['total_posts'].sum():,}")
    print(f"Total Comments: {stats['total_comments'].sum():,}")
    print(f"Total Data Points: {stats['total_posts'].sum() + stats['total_comments'].sum():,}")
    
    print("\n📋 By Subreddit:")
    for _, row in stats.iterrows():
        print(f"  r/{row['subreddit']}: {int(row['total_posts']):,} posts, {int(row['total_comments']):,} comments")
    
    conn.close()
    return stats

if __name__ == "__main__":
    create_collection_dashboard()
