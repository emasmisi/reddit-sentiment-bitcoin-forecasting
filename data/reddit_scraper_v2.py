import praw
import pandas as pd
import datetime
import time
import os
import json
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditScraperV2:
    def __init__(self):
        """Inizializza il scraper Reddit con credenziali da .env"""
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Crea cartelle se non esistono
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        logger.info(f"✅ Reddit configurato: {self.reddit.config.user_agent}")
        
    def scrape_subreddit(self, subreddit_name: str, post_limit: int = 500, 
                        days_back: int = 7, include_comments: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Scrapa un subreddit con filtri intelligenti
        
        Args:
            subreddit_name: Nome del subreddit
            post_limit: Limite massimo di post
            days_back: Giorni indietro da considerare
            include_comments: Se includere i commenti (max 20 per post)
        """
        logger.info(f"🚀 Iniziando scraping di r/{subreddit_name}")
        logger.info(f"   Parametri: {post_limit} post, ultimi {days_back} giorni")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Calcola timestamp di cutoff
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        cutoff_timestamp = cutoff_date.timestamp()
        
        posts_data = []
        comments_data = []
        
        count = 0
        skipped_old = 0
        
        try:
            # Mix di hot, new e top per diversità
            logger.info("📥 Raccogliendo post da hot, new e top...")
            
            submissions = []
            submissions.extend(list(subreddit.hot(limit=post_limit//2)))
            submissions.extend(list(subreddit.new(limit=post_limit//3)))
            submissions.extend(list(subreddit.top(time_filter='week', limit=post_limit//6)))
            
            # Rimuovi duplicati mantenendo l'ordine
            seen = set()
            unique_submissions = []
            for sub in submissions:
                if sub.id not in seen:
                    seen.add(sub.id)
                    unique_submissions.append(sub)
            
            logger.info(f"📊 Trovati {len(unique_submissions)} post unici da processare")
            
            for submission in unique_submissions:
                if count >= post_limit:
                    break
                    
                # Filtra per data
                if submission.created_utc < cutoff_timestamp:
                    skipped_old += 1
                    continue
                
                try:
                    # Filtra post troppo corti o spam
                    text_content = f"{submission.title} {submission.selftext}".strip()
                    if len(text_content) < 20:  # Post troppo corti
                        continue
                    
                    # Dati del post con campi aggiuntivi per analisi
                    post_data = {
                        'post_id': submission.id,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'text_content': text_content,  # Combinato per sentiment
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        'created_date': datetime.datetime.utcfromtimestamp(submission.created_utc).date().isoformat(),
                        'created_hour': datetime.datetime.utcfromtimestamp(submission.created_utc).hour,
                        'url': submission.url,
                        'subreddit': subreddit_name,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'is_self': submission.is_self,
                        'link_flair_text': submission.link_flair_text or '',
                        'over_18': submission.over_18
                    }
                    posts_data.append(post_data)
                    
                    # Scraping commenti se richiesto
                    if include_comments and submission.num_comments > 0:
                        comment_count = 0
                        try:
                            submission.comments.replace_more(limit=3)  # Limite per performance
                            
                            for comment in submission.comments.list():
                                if comment_count >= 20:  # Max 20 commenti per post
                                    break
                                    
                                if (hasattr(comment, 'body') and 
                                    comment.body not in ['[deleted]', '[removed]'] and
                                    len(comment.body.strip()) > 10):  # Commenti significativi
                                    
                                    comment_data = {
                                        'comment_id': comment.id,
                                        'post_id': submission.id,
                                        'comment_body': comment.body,
                                        'score': comment.score,
                                        'created_utc': datetime.datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                                        'created_date': datetime.datetime.utcfromtimestamp(comment.created_utc).date().isoformat(),
                                        'author': str(comment.author) if comment.author else '[deleted]',
                                        'is_submitter': comment.is_submitter,
                                        'parent_id': comment.parent_id
                                    }
                                    comments_data.append(comment_data)
                                    comment_count += 1
                                    
                        except Exception as e:
                            logger.warning(f"⚠️ Errore nei commenti del post {submission.id}: {e}")
                    
                    count += 1
                    if count % 25 == 0:
                        logger.info(f"📈 Progresso: {count} post processati, {len(comments_data)} commenti, {skipped_old} post vecchi saltati")
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logger.error(f"❌ Errore nel post {submission.id}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Errore generale: {e}")
        
        # Converti in DataFrame
        posts_df = pd.DataFrame(posts_data)
        comments_df = pd.DataFrame(comments_data)
        
        # Aggiungi statistiche di base
        if len(posts_df) > 0:
            posts_df['title_length'] = posts_df['title'].str.len()
            posts_df['text_length'] = posts_df['text_content'].str.len()
            posts_df['engagement_ratio'] = posts_df['num_comments'] / (posts_df['score'] + 1)
        
        logger.info(f"✅ Completato r/{subreddit_name}!")
        logger.info(f"   📊 {len(posts_df)} posts finali")
        logger.info(f"   💬 {len(comments_df)} commenti")
        logger.info(f"   📅 Periodo: {cutoff_date.date()} - {datetime.date.today()}")
        
        return {
            'posts': posts_df,
            'comments': comments_df
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], subreddit_name: str):
        """Salva i dati con timestamp e metadati"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        posts_csv = f"data/raw/{subreddit_name}_posts_{timestamp}.csv"
        comments_csv = f"data/raw/{subreddit_name}_comments_{timestamp}.csv"
        metadata_json = f"data/raw/{subreddit_name}_metadata_{timestamp}.json"
        
        # Salva CSV
        data['posts'].to_csv(posts_csv, index=False, encoding='utf-8')
        data['comments'].to_csv(comments_csv, index=False, encoding='utf-8')
        
        # Statistiche per metadati
        stats = {
            'subreddit': subreddit_name,
            'timestamp': timestamp,
            'posts_count': len(data['posts']),
            'comments_count': len(data['comments']),
            'date_range': {
                'start': data['posts']['created_date'].min() if len(data['posts']) > 0 else None,
                'end': data['posts']['created_date'].max() if len(data['posts']) > 0 else None
            },
            'files': {
                'posts': posts_csv,
                'comments': comments_csv
            }
        }
        
        if len(data['posts']) > 0:
            stats['posts_stats'] = {
                'avg_score': float(data['posts']['score'].mean()),
                'avg_comments': float(data['posts']['num_comments'].mean()),
                'avg_upvote_ratio': float(data['posts']['upvote_ratio'].mean())
            }
        
        # Salva metadati
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Dati salvati:")
        logger.info(f"   📄 Posts: {posts_csv}")
        logger.info(f"   💬 Comments: {comments_csv}")
        logger.info(f"   📋 Metadata: {metadata_json}")
        
        return stats

def main():
    """Funzione principale per scraping multipli subreddit"""
    
    # Configurazione
    SUBREDDITS = ['wallstreetbets', 'CryptoCurrency']  # Inizia con 2 per test
    POST_LIMIT = 300  # Numero ragionevole per iniziare
    DAYS_BACK = 7     # Ultima settimana
    
    logger.info("🚀 === AVVIO SCRAPING REDDIT PER TESI ===")
    logger.info(f"Subreddit target: {SUBREDDITS}")
    logger.info(f"Limit per subreddit: {POST_LIMIT} post")
    logger.info(f"Periodo: ultimi {DAYS_BACK} giorni")
    
    scraper = RedditScraperV2()
    results = {}
    
    for i, subreddit_name in enumerate(SUBREDDITS, 1):
        logger.info(f"\n🎯 === SUBREDDIT {i}/{len(SUBREDDITS)}: r/{subreddit_name} ===")
        
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
            results[subreddit_name] = metadata
            
            logger.info(f"✅ {subreddit_name} completato con successo!")
            
        except Exception as e:
            logger.error(f"❌ Errore con r/{subreddit_name}: {e}")
            results[subreddit_name] = {'error': str(e)}
        
        # Pausa tra subreddit per rispetto rate limits
        if i < len(SUBREDDITS):
            logger.info("⏸️ Pausa 10 secondi...")
            time.sleep(10)
    
    # Riassunto finale
    logger.info("\n🏁 === SCRAPING COMPLETATO ===")
    total_posts = sum(r.get('posts_count', 0) for r in results.values())
    total_comments = sum(r.get('comments_count', 0) for r in results.values())
    
    logger.info(f"📊 TOTALI:")
    logger.info(f"   Subreddit processati: {len(results)}")
    logger.info(f"   Post totali: {total_posts}")
    logger.info(f"   Commenti totali: {total_comments}")
    
    for subreddit, stats in results.items():
        if 'error' in stats:
            logger.error(f"   ❌ r/{subreddit}: ERRORE - {stats['error']}")
        else:
            logger.info(f"   ✅ r/{subreddit}: {stats['posts_count']} posts, {stats['comments_count']} commenti")

if __name__ == "__main__":
    main()