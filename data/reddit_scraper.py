import praw
import pandas as pd
import datetime
import time
import os
import json
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Inizializza il scraper Reddit"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Crea cartella data se non esiste
        os.makedirs('data/raw', exist_ok=True)
        
    def scrape_subreddit(self, subreddit_name: str, post_limit: int = 1000, 
                        days_back: int = 30, include_comments: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Scrapa un subreddit con filtri temporali
        
        Args:
            subreddit_name: Nome del subreddit
            post_limit: Limite massimo di post
            days_back: Giorni indietro da considerare
            include_comments: Se includere i commenti
        
        Returns:
            Dict con DataFrame di posts e comments
        """
        logger.info(f"Iniziando scraping di r/{subreddit_name}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Calcola timestamp di cutoff
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        cutoff_timestamp = cutoff_date.timestamp()
        
        posts_data = []
        comments_data = []
        
        count = 0
        skipped_old = 0
        
        try:
            # Combina hot, new e top per diversità
            submissions = list(subreddit.hot(limit=post_limit//3)) + \
                         list(subreddit.new(limit=post_limit//3)) + \
                         list(subreddit.top(time_filter='month', limit=post_limit//3))
            
            # Rimuovi duplicati
            unique_submissions = {sub.id: sub for sub in submissions}.values()
            
            for submission in unique_submissions:
                if count >= post_limit:
                    break
                    
                # Filtra per data
                if submission.created_utc < cutoff_timestamp:
                    skipped_old += 1
                    continue
                
                try:
                    # Dati del post
                    post_data = {
                        'post_id': submission.id,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        'url': submission.url,
                        'subreddit': subreddit_name,
                        'author': str(submission.author) if submission.author else '[deleted]'
                    }
                    posts_data.append(post_data)
                    
                    # Scraping commenti se richiesto
                    if include_comments and submission.num_comments > 0:
                        try:
                            submission.comments.replace_more(limit=5)  # Limite per evitare troppi commenti
                            
                            for comment in submission.comments.list()[:50]:  # Max 50 commenti per post
                                if hasattr(comment, 'body') and comment.body != '[deleted]':
                                    comment_data = {
                                        'comment_id': comment.id,
                                        'post_id': submission.id,
                                        'comment_body': comment.body,
                                        'score': comment.score,
                                        'created_utc': datetime.datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                                        'author': str(comment.author) if comment.author else '[deleted]'
                                    }
                                    comments_data.append(comment_data)
                                    
                        except Exception as e:
                            logger.warning(f"Errore nei commenti del post {submission.id}: {e}")
                    
                    count += 1
                    if count % 50 == 0:
                        logger.info(f"Scaricati {count} post, saltati {skipped_old} post vecchi")
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logger.error(f"Errore nel post {submission.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Errore generale: {e}")
        
        # Converti in DataFrame
        posts_df = pd.DataFrame(posts_data)
        comments_df = pd.DataFrame(comments_data)
        
        logger.info(f"Completato! {len(posts_df)} posts, {len(comments_df)} commenti")
        
        return {
            'posts': posts_df,
            'comments': comments_df
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], subreddit_name: str):
        """Salva i dati in formato CSV e JSON"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva CSV
        posts_csv = f"data/raw/{subreddit_name}_posts_{timestamp}.csv"
        comments_csv = f"data/raw/{subreddit_name}_comments_{timestamp}.csv"
        
        data['posts'].to_csv(posts_csv, index=False, encoding='utf-8')
        data['comments'].to_csv(comments_csv, index=False, encoding='utf-8')
        
        # Salva anche metadati
        metadata = {
            'subreddit': subreddit_name,
            'timestamp': timestamp,
            'posts_count': len(data['posts']),
            'comments_count': len(data['comments']),
            'files': {
                'posts': posts_csv,
                'comments': comments_csv
            }
        }
        
        with open(f"data/raw/{subreddit_name}_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dati salvati: {posts_csv}, {comments_csv}")
        return metadata

def main():
    # --- Configurazione (AGGIORNA QUESTI VALORI) ---
    CLIENT_ID = 'TUO_CLIENT_ID'
    CLIENT_SECRET = 'TUO_CLIENT_SECRET'
    USER_AGENT = 'script:tesi_sentiment:v1.0 (by /u/tuo_username)'
    
    # Configurazione scraping
    SUBREDDITS = ['wallstreetbets', 'CryptoCurrency', 'investing']  # Multipli subreddit
    POST_LIMIT = 1000  # Per subreddit
    DAYS_BACK = 30  # Ultimi 30 giorni
    
    scraper = RedditScraper(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    for subreddit_name in SUBREDDITS:
        logger.info(f"=== Scraping r/{subreddit_name} ===")
        
        try:
            # Scraping
            data = scraper.scrape_subreddit(
                subreddit_name=subreddit_name,
                post_limit=POST_LIMIT,
                days_back=DAYS_BACK,
                include_comments=True
            )
            
            # Salvataggio
            metadata = scraper.save_data(data, subreddit_name)
            
            logger.info(f"Completato {subreddit_name}: {metadata['posts_count']} posts, {metadata['comments_count']} commenti")
            
        except Exception as e:
            logger.error(f"Errore con r/{subreddit_name}: {e}")
        
        # Pausa tra subreddit
        time.sleep(5)

if __name__ == "__main__":
    main()
