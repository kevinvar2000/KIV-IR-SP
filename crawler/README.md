# Web Crawler – interez.sk

A Python web crawler that systematically crawls the [interez.sk](https://www.interez.sk/) website, extracts page metadata, and stores the results in JSON format.

## Features

- **robots.txt compliance** – Parses `robots.txt` to respect `Allow`, `Disallow`, and `Crawl-delay` directives.
- **Sitemap support** – Fetches and recursively processes XML sitemaps to discover URLs.
- **State persistence** – Saves crawler state (`crawler_state.json`) so crawling can be resumed after interruption.
- **Metadata extraction** – Extracts page title, author, publication date, and a content hash (MD5) for each crawled page.
- **Polite crawling** – Configurable request delay between requests with proper `User-Agent` headers.
- **Duplicate detection** – Tracks visited and pending URLs to avoid re-crawling.

## Requirements

- Python 3.x
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
- [Requests](https://pypi.org/project/requests/)

### Install dependencies

```bash
pip install beautifulsoup4 requests
```

## Usage

```bash
python crawler.py
```

The crawler will:

1. Attempt to load a previously saved state from `crawler_state.json`.
2. If no saved state exists, fetch `robots.txt` and the sitemap to seed the URL queue.
3. Crawl pages from the pending URL queue, extracting metadata and discovering new links.
4. Save progress after each page so it can be resumed later.

## Output Files

| File | Description |
|---|---|
| `crawled_pages.json` | One JSON object per line containing extracted metadata for each crawled page. |
| `crawler_state.json` | Crawler state (pending URLs, visited URLs, configuration) for resuming. |

### Crawled page entry format

```json
{
  "url": "https://www.interez.sk/example-article/",
  "title": "Example Article Title",
  "author": "Author Name",
  "topic": "Article's Topic",
  "publication_date": "2025-01-15T10:00:00+00:00",
  "hashed_content": "d41d8cd98f00b204e9800998ecf8427e",
  "article_text": "Article Text",
  "scraped_at": "2025-01-20 14:30:00"
}
```

## Configuration

Key constants at the top of `crawler.py`:

| Constant | Default | Description |
|---|---|---|
| `INIT_URL` | `https://www.interez.sk/` | Base URL – only links under this domain are followed. |
| `REQUEST_DELAY` | `3` (seconds) | Delay between HTTP requests (may be overridden by `Crawl-delay` in robots.txt). |
| `MAX_URLS` | `1000` | Maximum URL limit (defined but not yet enforced). |
| `DISALLOWED_PATHS` | `/profil/`, `/kontakt/`, `/o-nas/` | Default paths to skip (extended by robots.txt). |
