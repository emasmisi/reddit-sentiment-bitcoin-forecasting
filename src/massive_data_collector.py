import praw
import sqlite3
import pandas as pd
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "data/reddit_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Posts table (FIXED)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                subreddit TEXT NOT NULL,
                title TEXT NOT NULL,
                selftext TEXT,
                score INTEGER,
                upvote_ratio REAL,
                num_comments INTEGER,
                created_utc TEXT,
                created_date TEXT,
                created_hour INTEGER,
                author TEXT,
                url TEXT,
                text_content TEXT,
                collection_timestamp TEXT
            )
        ''')
        
        # Comments table (FIXED)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                subreddit TEXT NOT NULL,
                comment_body TEXT NOT NULL,
                score INTEGER,
                created_utc TEXT,
                created_date TEXT,
                author TEXT,
                collection_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def insert_post(self, post_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_data['post_id'], post_data['subreddit'], post_data['title'],
                post_data['selftext'], post_data['score'], post_data['upvote_ratio'],
                post_data['num_comments'], post_data['created_utc'], post_data['created_date'],
                post_data['created_hour'], post_data['author'], post_data['url'],
                post_data['text_content'], post_data['collection_timestamp']
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error inserting post: {e}")
            return False
    
    def insert_comment(self, comment_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                comment_data['comment_id'], comment_data['post_id'], comment_data['subreddit'],
                comment_data['comment_body'], comment_data['score'], comment_data['created_utc'],
                comment_data['created_date'], comment_data['author'], comment_data['collection_timestamp']
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error inserting comment: {e}")
            return False

class SimpleRedditCollector:
    def __init__(self):
        self.db = DatabaseManager()
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        logger.info("✅ Reddit API initialized")
    
    def collect_subreddit(self, subreddit_name: str, target_posts: int = 1000):
        logger.info(f"🎯 Starting collection for r/{subreddit_name}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        collected_posts = 0
        collected_comments = 0
        
        try:
            for submission in subreddit.hot(limit=target_posts * 2):
                if collected_posts >= target_posts:
                    break
                
                # Process post
                text_content = f"{submission.title} {submission.selftext}".strip()
                if len(text_content) < 20:
                    continue
                
                post_data = {
                    'post_id': submission.id,
                    'subreddit': subreddit_name,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    'created_date': datetime.utcfromtimestamp(submission.created_utc).date().isoformat(),
                    'created_hour': datetime.utcfromtimestamp(submission.created_utc).hour,
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'url': submission.url,
                    'text_content': text_content,
                    'collection_timestamp': datetime.now().isoformat()
                }
                
                if self.db.insert_post(post_data):
                    collected_posts += 1
                    
                    # Process comments
                    try:
                        submission.comments.replace_more(limit=3)
                        for comment in submission.comments.list()[:15]:
                            if hasattr(comment, 'body') and len(comment.body) > 10:
                                comment_data = {
                                    'comment_id': comment.id,
                                    'post_id': submission.id,
                                    'subreddit': subreddit_name,
                                    'comment_body': comment.body,
                                    'score': comment.score,
                                    'created_utc': datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                                    'created_date': datetime.utcfromtimestamp(comment.created_utc).date().isoformat(),
                                    'author': str(comment.author) if comment.author else '[deleted]',
                                    'collection_timestamp': datetime.now().isoformat()
                                }
                                if self.db.insert_comment(comment_data):
                                    collected_comments += 1
                    except:
                        pass
                
                if collected_posts % 25 == 0:
                    logger.info(f"📈 Progress: {collected_posts} posts, {collected_comments} comments")
                
                time.sleep(2)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error in collection: {e}")
        
        logger.info(f"✅ Completed r/{subreddit_name}: {collected_posts} posts, {collected_comments} comments")
        return collected_posts, collected_comments

def main():
    collector = SimpleRedditCollector()
    
    subreddits = ['wallstreetbets', 'CryptoCurrency', 'investing', 'stocks']
    target_per_subreddit = 1000
    
    total_posts = 0
    total_comments = 0
    
    for subreddit in subreddits:
        posts, comments = collector.collect_subreddit(subreddit, target_per_subreddit)
        total_posts += posts
        total_comments += comments
        
        logger.info(f"⏸️ Cool-down 30 seconds...")
        time.sleep(30)
    
    logger.info(f"🎉 COLLECTION COMPLETED!")
    logger.info(f"📊 Total: {total_posts} posts, {total_comments} comments")

if __name__ == "__main__":
    main()