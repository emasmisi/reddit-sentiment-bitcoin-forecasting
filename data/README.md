# data/

Raw and processed data are **not committed**. The scripts in `src/` create:

- `data/financial/`: hourly OHLCV (BTC, ETH, S&P 500, NASDAQ, VIX, DXY)
- `data/raw/`: Reddit posts/comments per subreddit
- `data/processed/`: sentiment scores, aligned datasets, correlation outputs
- `data/reddit_data.db`: SQLite store used by the collector
