#!/usr/bin/env python3
"""
📖 REDDIT TEXT CONTENT VIEWER
=============================
Visualizzatore interattivo per esplorare i testi dei post e commenti raccolti
- Browse posts per subreddit
- Ricerca per parole chiave
- Filtri per score/engagement
- Preview testo completo
"""

import pandas as pd
import sqlite3
import random
from typing import List, Dict, Optional
import re

class RedditTextViewer:
    """📖 Visualizzatore per contenuti Reddit raccolti"""
    
    def __init__(self, db_path: str = "data/reddit_data.db"):
        self.db_path = db_path
        print("📊 === REDDIT TEXT CONTENT VIEWER ===")
        print("Caricamento dati dal database...")
        self.load_data()
        self.show_overview()
        
    def load_data(self):
        """📥 Carica dati dal database"""
        conn = sqlite3.connect(self.db_path)
        
        # Carica posts
        posts_query = """
        SELECT 
            post_id, subreddit, title, selftext, text_content,
            score, num_comments, created_date, created_hour, author
        FROM posts
        ORDER BY created_date DESC
        """
        self.posts_df = pd.read_sql_query(posts_query, conn)
        
        # Carica comments  
        comments_query = """
        SELECT 
            comment_id, post_id, subreddit, comment_body,
            score, created_date, author
        FROM comments
        ORDER BY created_date DESC
        """
        self.comments_df = pd.read_sql_query(comments_query, conn)
        
        conn.close()
        
        # Preprocessing
        self.posts_df['full_text'] = (
            self.posts_df['title'].fillna('') + ' ' + 
            self.posts_df['selftext'].fillna('') + ' ' +
            self.posts_df['text_content'].fillna('')
        ).str.strip()
        
        self.posts_df['text_length'] = self.posts_df['full_text'].str.len()
        self.comments_df['text_length'] = self.comments_df['comment_body'].str.len()
        
        print(f"✅ Caricati {len(self.posts_df)} posts e {len(self.comments_df)} comments")
        
    def show_overview(self):
        """📊 Mostra overview generale"""
        print(f"\n📈 === OVERVIEW DATASET ===")
        print(f"📋 Posts totali: {len(self.posts_df):,}")
        print(f"📋 Comments totali: {len(self.comments_df):,}")
        
        print(f"\n📊 Posts per subreddit:")
        for subreddit, count in self.posts_df['subreddit'].value_counts().items():
            print(f"   r/{subreddit}: {count:,} posts")
            
        print(f"\n📊 Lunghezza media testi:")
        print(f"   Posts: {self.posts_df['text_length'].mean():.0f} caratteri")
        print(f"   Comments: {self.comments_df['text_length'].mean():.0f} caratteri")
        
    def browse_posts(self, subreddit: Optional[str] = None, limit: int = 10, 
                    min_score: int = 0, random_sample: bool = False):
        """🔍 Naviga posts con filtri"""
        
        df = self.posts_df.copy()
        
        # Filtri
        if subreddit:
            df = df[df['subreddit'] == subreddit]
            print(f"🎯 Filtrando per r/{subreddit}")
            
        if min_score > 0:
            df = df[df['score'] >= min_score]
            print(f"📈 Filtrando per score >= {min_score}")
            
        if len(df) == 0:
            print("❌ Nessun post trovato con questi filtri!")
            return
            
        # Sampling
        if random_sample:
            df = df.sample(min(limit, len(df)))
            print(f"🎲 Sample casuale di {len(df)} posts")
        else:
            df = df.head(limit)
            print(f"📝 Top {len(df)} posts")
            
        print(f"\n{'='*80}")
        
        for i, (_, post) in enumerate(df.iterrows(), 1):
            self._display_post(post, i)
            print(f"{'-'*60}")
            
    def browse_comments(self, subreddit: Optional[str] = None, limit: int = 10,
                       min_score: int = 0, random_sample: bool = False):
        """💬 Naviga comments con filtri"""
        
        df = self.comments_df.copy()
        
        # Filtri
        if subreddit:
            df = df[df['subreddit'] == subreddit]
            print(f"🎯 Filtrando per r/{subreddit}")
            
        if min_score > 0:
            df = df[df['score'] >= min_score]
            print(f"📈 Filtrando per score >= {min_score}")
            
        if len(df) == 0:
            print("❌ Nessun comment trovato con questi filtri!")
            return
            
        # Sampling
        if random_sample:
            df = df.sample(min(limit, len(df)))
            print(f"🎲 Sample casuale di {len(df)} comments")
        else:
            df = df.head(limit)
            print(f"💬 Top {len(df)} comments")
            
        print(f"\n{'='*80}")
        
        for i, (_, comment) in enumerate(df.iterrows(), 1):
            self._display_comment(comment, i)
            print(f"{'-'*60}")
            
    def search_posts(self, keywords: List[str], subreddit: Optional[str] = None, 
                    limit: int = 10):
        """🔍 Ricerca posts per parole chiave"""
        
        df = self.posts_df.copy()
        
        if subreddit:
            df = df[df['subreddit'] == subreddit]
            
        # Ricerca nelle colonne di testo
        pattern = '|'.join(keywords)
        mask = (
            df['title'].str.contains(pattern, case=False, na=False) |
            df['selftext'].str.contains(pattern, case=False, na=False) |
            df['text_content'].str.contains(pattern, case=False, na=False)
        )
        
        results = df[mask].head(limit)
        
        print(f"🔍 Ricerca per: {', '.join(keywords)}")
        if subreddit:
            print(f"🎯 In r/{subreddit}")
        print(f"📊 Trovati {len(results)} risultati")
        
        if len(results) == 0:
            print("❌ Nessun risultato trovato!")
            return
            
        print(f"\n{'='*80}")
        
        for i, (_, post) in enumerate(results.iterrows(), 1):
            self._display_post(post, i, highlight_keywords=keywords)
            print(f"{'-'*60}")
            
    def search_comments(self, keywords: List[str], subreddit: Optional[str] = None,
                       limit: int = 10):
        """🔍 Ricerca comments per parole chiave"""
        
        df = self.comments_df.copy()
        
        if subreddit:
            df = df[df['subreddit'] == subreddit]
            
        # Ricerca nel corpo del commento
        pattern = '|'.join(keywords)
        mask = df['comment_body'].str.contains(pattern, case=False, na=False)
        
        results = df[mask].head(limit)
        
        print(f"🔍 Ricerca per: {', '.join(keywords)}")
        if subreddit:
            print(f"🎯 In r/{subreddit}")
        print(f"📊 Trovati {len(results)} risultati")
        
        if len(results) == 0:
            print("❌ Nessun risultato trovato!")
            return
            
        print(f"\n{'='*80}")
        
        for i, (_, comment) in enumerate(results.iterrows(), 1):
            self._display_comment(comment, i, highlight_keywords=keywords)
            print(f"{'-'*60}")
            
    def get_post_details(self, post_id: str):
        """📄 Mostra dettagli completi di un post"""
        
        post = self.posts_df[self.posts_df['post_id'] == post_id]
        
        if len(post) == 0:
            print(f"❌ Post {post_id} non trovato!")
            return
            
        post = post.iloc[0]
        
        print(f"\n{'='*80}")
        print(f"📄 DETTAGLI POST COMPLETI")
        print(f"{'='*80}")
        
        self._display_post(post, detailed=True)
        
        # Mostra commenti del post
        post_comments = self.comments_df[self.comments_df['post_id'] == post_id]
        
        if len(post_comments) > 0:
            print(f"\n💬 COMMENTI ({len(post_comments)}):")
            print(f"{'-'*40}")
            
            for i, (_, comment) in enumerate(post_comments.head(10).iterrows(), 1):
                self._display_comment(comment, i, compact=True)
                
            if len(post_comments) > 10:
                print(f"... e altri {len(post_comments)-10} commenti")
        else:
            print(f"\n💬 Nessun commento trovato per questo post")
            
    def show_stats(self, subreddit: Optional[str] = None):
        """📊 Mostra statistiche dettagliate"""
        
        posts_df = self.posts_df if not subreddit else self.posts_df[self.posts_df['subreddit'] == subreddit]
        comments_df = self.comments_df if not subreddit else self.comments_df[self.comments_df['subreddit'] == subreddit]
        
        title = f"📊 STATISTICHE {'GENERALI' if not subreddit else f'r/{subreddit.upper()}'}"
        print(f"\n{title}")
        print(f"{'='*len(title)}")
        
        # Posts stats
        print(f"📝 POSTS:")
        print(f"   Totali: {len(posts_df):,}")
        print(f"   Score medio: {posts_df['score'].mean():.1f}")
        print(f"   Commenti medi: {posts_df['num_comments'].mean():.1f}")
        print(f"   Lunghezza media: {posts_df['text_length'].mean():.0f} caratteri")
        
        # Comments stats
        print(f"\n💬 COMMENTS:")
        print(f"   Totali: {len(comments_df):,}")
        print(f"   Score medio: {comments_df['score'].mean():.1f}")
        print(f"   Lunghezza media: {comments_df['text_length'].mean():.0f} caratteri")
        
        # Top posts
        print(f"\n🏆 TOP 5 POSTS PER SCORE:")
        top_posts = posts_df.nlargest(5, 'score')[['title', 'score', 'num_comments']]
        for i, (_, post) in enumerate(top_posts.iterrows(), 1):
            title = post['title'][:60] + '...' if len(post['title']) > 60 else post['title']
            print(f"   {i}. {title} (Score: {post['score']}, Comments: {post['num_comments']})")
            
    def _display_post(self, post, index: int = 1, detailed: bool = False, 
                     highlight_keywords: List[str] = None):
        """📄 Display singolo post"""
        
        print(f"📄 POST #{index} | r/{post['subreddit']} | Score: {post['score']} | Comments: {post['num_comments']}")
        print(f"👤 Author: {post['author']} | 📅 Date: {post['created_date']}")
        
        title = post['title'] if pd.notna(post['title']) else '[No Title]'
        if highlight_keywords:
            title = self._highlight_text(title, highlight_keywords)
        print(f"📌 TITLE: {title}")
        
        if detailed or pd.notna(post['selftext']) and len(str(post['selftext']).strip()) > 0:
            selftext = str(post['selftext']) if pd.notna(post['selftext']) else ''
            if highlight_keywords:
                selftext = self._highlight_text(selftext, highlight_keywords)
            if len(selftext) > 500 and not detailed:
                selftext = selftext[:500] + '...'
            if selftext.strip():
                print(f"📝 CONTENT: {selftext}")
                
        if detailed:
            print(f"🔗 ID: {post['post_id']}")
            print(f"📊 Text Length: {post['text_length']} characters")
            
    def _display_comment(self, comment, index: int = 1, compact: bool = False,
                        highlight_keywords: List[str] = None):
        """💬 Display singolo comment"""
        
        if not compact:
            print(f"💬 COMMENT #{index} | r/{comment['subreddit']} | Score: {comment['score']}")
            print(f"👤 Author: {comment['author']} | 📅 Date: {comment['created_date']}")
            
        body = comment['comment_body']
        if highlight_keywords:
            body = self._highlight_text(body, highlight_keywords)
            
        if len(body) > 300 and compact:
            body = body[:300] + '...'
            
        print(f"💭 {body}")
        
    def _highlight_text(self, text: str, keywords: List[str]) -> str:
        """🔍 Evidenzia keywords nel testo"""
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(f"**{keyword.upper()}**", text)
        return text
        
    def interactive_menu(self):
        """🎮 Menu interattivo"""
        
        while True:
            print(f"\n🎮 === REDDIT TEXT VIEWER - MENU INTERATTIVO ===")
            print("1. 📝 Browse Posts")
            print("2. 💬 Browse Comments") 
            print("3. 🔍 Ricerca Posts")
            print("4. 🔍 Ricerca Comments")
            print("5. 📄 Dettagli Post Specifico")
            print("6. 📊 Statistiche")
            print("7. 🎲 Posts Casuali")
            print("8. 🎲 Comments Casuali")
            print("0. 🚪 Esci")
            
            choice = input("\n👉 Scegli opzione (0-8): ").strip()
            
            try:
                if choice == '0':
                    print("👋 Arrivederci!")
                    break
                elif choice == '1':
                    self._interactive_browse_posts()
                elif choice == '2':
                    self._interactive_browse_comments()
                elif choice == '3':
                    self._interactive_search_posts()
                elif choice == '4':
                    self._interactive_search_comments()
                elif choice == '5':
                    self._interactive_post_details()
                elif choice == '6':
                    self._interactive_stats()
                elif choice == '7':
                    self._interactive_random_posts()
                elif choice == '8':
                    self._interactive_random_comments()
                else:
                    print("❌ Opzione non valida!")
                    
            except KeyboardInterrupt:
                print("\n👋 Operazione interrotta!")
                break
            except Exception as e:
                print(f"❌ Errore: {e}")
                
    def _interactive_browse_posts(self):
        """🔍 Browse posts interattivo"""
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        limit = int(input("📊 Limite (default 10): ").strip() or "10")
        min_score = int(input("📈 Score minimo (default 0): ").strip() or "0")
        
        self.browse_posts(subreddit=subreddit, limit=limit, min_score=min_score)
        
    def _interactive_browse_comments(self):
        """💬 Browse comments interattivo"""
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        limit = int(input("📊 Limite (default 10): ").strip() or "10")
        min_score = int(input("📈 Score minimo (default 0): ").strip() or "0")
        
        self.browse_comments(subreddit=subreddit, limit=limit, min_score=min_score)
        
    def _interactive_search_posts(self):
        """🔍 Ricerca posts interattiva"""
        keywords_input = input("🔍 Parole chiave (separate da virgola): ").strip()
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        
        if not keywords:
            print("❌ Inserisci almeno una parola chiave!")
            return
            
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        limit = int(input("📊 Limite (default 10): ").strip() or "10")
        
        self.search_posts(keywords=keywords, subreddit=subreddit, limit=limit)
        
    def _interactive_search_comments(self):
        """🔍 Ricerca comments interattiva"""
        keywords_input = input("🔍 Parole chiave (separate da virgola): ").strip()
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        
        if not keywords:
            print("❌ Inserisci almeno una parola chiave!")
            return
            
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        limit = int(input("📊 Limite (default 10): ").strip() or "10")
        
        self.search_comments(keywords=keywords, subreddit=subreddit, limit=limit)
        
    def _interactive_post_details(self):
        """📄 Dettagli post interattivo"""
        post_id = input("📄 Post ID: ").strip()
        
        if not post_id:
            print("❌ Inserisci un Post ID!")
            return
            
        self.get_post_details(post_id)
        
    def _interactive_stats(self):
        """📊 Statistiche interattive"""
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        self.show_stats(subreddit=subreddit)
        
    def _interactive_random_posts(self):
        """🎲 Posts casuali interattivi"""
        limit = int(input("📊 Quanti post casuali? (default 5): ").strip() or "5")
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        
        self.browse_posts(subreddit=subreddit, limit=limit, random_sample=True)
        
    def _interactive_random_comments(self):
        """🎲 Comments casuali interattivi"""
        limit = int(input("📊 Quanti comments casuali? (default 5): ").strip() or "5")
        subreddit = input("🎯 Subreddit (lascia vuoto per tutti): ").strip() or None
        
        self.browse_comments(subreddit=subreddit, limit=limit, random_sample=True)

def main():
    """🚀 Main function"""
    viewer = RedditTextViewer()
    
    print(f"\n🎮 === ESEMPI DI UTILIZZO ===")
    print("viewer.browse_posts(subreddit='wallstreetbets', limit=5)")
    print("viewer.search_posts(['TSLA', 'Tesla'], limit=10)")
    print("viewer.show_stats('CryptoCurrency')")
    print("viewer.interactive_menu()  # <-- Menu interattivo!")
    
    # Avvia menu interattivo
    viewer.interactive_menu()

if __name__ == "__main__":
    main()