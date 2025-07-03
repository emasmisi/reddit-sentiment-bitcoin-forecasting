import praw
import os
from dotenv import load_dotenv

# Carica variabili da .env
load_dotenv()

def test_reddit_connection():
    """Test delle credenziali Reddit API"""
    
    try:
        # Configura Reddit
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        print("🔄 Testando connessione Reddit API...")
        
        # Test 1: Verifica che Reddit sia configurato correttamente
        print(f"✅ Reddit configurato: {reddit.config.user_agent}")
        
        # Test 2: Accesso a un subreddit pubblico
        subreddit = reddit.subreddit('wallstreetbets')
        print(f"✅ Accesso a r/wallstreetbets: {subreddit.display_name}")
        print(f"   Subscribers: {subreddit.subscribers:,}")
        
        # Test 3: Scarica 3 post di prova
        print("\n📝 Test scaricamento post:")
        post_count = 0
        for submission in subreddit.hot(limit=3):
            post_count += 1
            print(f"   Post {post_count}: {submission.title[:50]}...")
            print(f"   Score: {submission.score}, Comments: {submission.num_comments}")
        
        print(f"\n🎉 SUCCESSO! API Reddit funziona correttamente")
        print(f"   - Connessione: ✅")
        print(f"   - Accesso subreddit: ✅") 
        print(f"   - Download post: ✅ ({post_count} post)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        print("\n🔧 Possibili soluzioni:")
        print("   1. Verifica CLIENT_ID e CLIENT_SECRET nel file .env")
        print("   2. Controlla che l'app Reddit sia di tipo 'script'")
        print("   3. Assicurati che le credenziali siano corrette")
        return False

if __name__ == "__main__":
    # Verifica che il file .env esista
    if not os.path.exists('.env'):
        print("❌ File .env non trovato!")
        print("   Crea il file .env con le tue credenziali Reddit")
        exit(1)
    
    # Verifica che le variabili siano impostate
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Variabili mancanti nel .env: {missing_vars}")
        exit(1)
    
    # Esegui test
    test_reddit_connection()