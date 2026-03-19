# import scrapy
import json
from bs4 import BeautifulSoup
import requests
import time
import hashlib
from pathlib import Path


INIT_URL = 'https://www.interez.sk/'
ROBOTS_URL = 'https://www.interez.sk/robots.txt'
SITEMAP_URL = 'https://www.interez.sk/sitemap.xml'

PENDING_URLS = []
VISITED_URLS = []
LAST_VISITED_URL = None
ALLOWED_PATHS = []
DISALLOWED_PATHS = ['/profil/', '/kontakt/', '/o-nas/']

MAX_URLS = 1000

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}
REQUEST_DELAY = 3  # seconds, maybe random between 1 and 3/5 seconds to be more polite

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data' / 'crawler'
STATE_FILE = DATA_DIR / 'crawler_state.json'
OUTPUT_FILE = DATA_DIR / 'crawled_pages.json'


def index_url(url):
    # Add the URL to the index (e.g., save to a database or file)
    pass


def extract_urls_from_sitemap(sitemap_content):
    urls = []
    for line in sitemap_content.splitlines():
        if line.strip().startswith('<loc>') and line.strip().endswith('</loc>'):
            url = line.strip()[5:-6]
            urls.append(url)

    return urls


def is_valid_url(url):
    return url.startswith(INIT_URL)


def is_allowed_by_robots(url):

    for disallowed_path in DISALLOWED_PATHS:
        if url.startswith(disallowed_path):
            return False
    return True


def save_state():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        state = {
            'pending_urls': PENDING_URLS,
            'visited_urls': VISITED_URLS,
            'allowed_paths': ALLOWED_PATHS,
            'disallowed_paths': DISALLOWED_PATHS,
            'request_delay': REQUEST_DELAY
        }
        json.dump(state, f, indent=4)


def load_state():
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            global PENDING_URLS, VISITED_URLS, ALLOWED_PATHS, DISALLOWED_PATHS, REQUEST_DELAY
            PENDING_URLS = state.get('pending_urls', [])
            VISITED_URLS = state.get('visited_urls', [])
            ALLOWED_PATHS = state.get('allowed_paths', [])
            DISALLOWED_PATHS = state.get('disallowed_paths', [])
            REQUEST_DELAY = state.get('request_delay', 2)
            print(f'Loaded crawler state: {len(PENDING_URLS)} pending URLs, {len(VISITED_URLS)} visited URLs')
            return True
    except FileNotFoundError:
        return False


def store_data(url, content):
    title = extract_title(content)
    author = extract_author(content)
    topic = extract_topic(content)
    publication_date = extract_publication_date(content)
    article_text = extract_article_text(content)

    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

    data = {
        'url': url,
        'title': title,
        'author': author,
        'topic': topic,
        'publication_date': publication_date,
        'hashed_content': content_hash,
        'article_text': article_text,
        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def extract_title(content):
    soup = BeautifulSoup(content, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text(strip=True)
    return "No Title Found"


def extract_author(content):
    soup = BeautifulSoup(content, 'html.parser')
    author_tag = soup.find('meta', attrs={'name': 'author'})
    if author_tag and 'content' in author_tag.attrs:
        return author_tag['content'].strip()
    return "No Author Found"


def extract_topic(content):
    soup = BeautifulSoup(content, 'html.parser')
    topic_tag = soup.find('a', class_='topic')
    if topic_tag:
        return topic_tag.get_text(strip=True)
    return "No Topic Found"


def extract_article_text(content):
    soup = BeautifulSoup(content, 'html.parser')
    article_tag = soup.find('article', id='clanok')
    if article_tag:
        return article_tag.get_text(strip=True)
    return "No Article Text Found"


def extract_publication_date(content):
    soup = BeautifulSoup(content, 'html.parser')
    pub_date_tag = soup.find('time', class_='entry-date')

    if pub_date_tag:
        if 'datetime' in pub_date_tag.attrs:
            return pub_date_tag['datetime'].strip()
        return pub_date_tag['content'].strip()
    return "No Publication Date Found"


def extract_urls_from_page(content):
    urls = []
    soup = BeautifulSoup(content, 'html.parser')
    for link in soup.find_all('a', href=True):
        url = link['href']
        if is_valid_url(url) and is_allowed_by_robots(url) and url not in VISITED_URLS and url not in PENDING_URLS and url not in urls:
            urls.append(url)
    print(f'Extracted {len(urls)} URLs from the page')
    PENDING_URLS.extend(urls)


def download_page(url):
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f'Failed to download {url}: {response.status_code}')
            return None
    except requests.RequestException as e:
        print(f'Error downloading {url}: {e}')
        return None


def crawl():
    while PENDING_URLS:

        url = PENDING_URLS.pop(0)
        if url in VISITED_URLS:
            print(f'Skipping already visited URL: {url}')
            continue

        if not is_valid_url(url):
            print(f'Skipping invalid URL: {url}')
            continue

        if not is_allowed_by_robots(url):
            print(f'Skipping disallowed URL: {url}')
            continue

        delay()

        content = download_page(url)
        if content is None:
            print(f'Skipping URL due to download failure: {url}')
            continue
        store_data(url, content)

        extract_urls_from_page(content)

        VISITED_URLS.append(url)
        print_progress()
        save_state()


def fetch_sitemap(sitemap_url):
    if sitemap_url is None:
        sitemap_url = SITEMAP_URL

    print(f'Fetching sitemap: {sitemap_url}')
    delay()

    response = requests.get(sitemap_url, headers=REQUEST_HEADERS, timeout=10)
    if response.status_code == 200:
        content = response.text

        new_urls = extract_urls_from_sitemap(content)
        for url in new_urls:
            if is_valid_url(url) and is_allowed_by_robots(url) and url not in PENDING_URLS:

                if 'sitemap'in url.lower():
                    fetch_sitemap(url)
                else:
                    PENDING_URLS.append(url)

        print(f'Initial URLs to crawl from sitemap: {len(PENDING_URLS)}')
        print_progress()
        save_state()
    else:
        print(f'Failed to fetch sitemap.xml: {response.status_code}')


def fetch_robots_txt():
    response = requests.get(ROBOTS_URL)
    if response.status_code == 200:
        content = response.text

        for line in content.splitlines():
            if line.startswith('Disallow:'):
                path = line.split(':')[1].strip()
                DISALLOWED_PATHS.append(path)
            
            elif line.startswith('Allow:'):
                path = line.split(':')[1].strip()
                ALLOWED_PATHS.append(path)

            elif line.startswith('Crawl-delay:'):
                delay = int(line.split(':')[1].strip())
                global REQUEST_DELAY
                REQUEST_DELAY = delay

            elif line.startswith('Sitemap:'):
                sitemap_url = line.split('Sitemap:')[1].strip()
                fetch_sitemap(sitemap_url)
    else:
        print(f'Failed to fetch robots.txt: {response.status_code}')


def delay():
    time.sleep(REQUEST_DELAY)


def print_progress():
    print(f'Visited URLs: {len(VISITED_URLS)} | Pending URLs: {len(PENDING_URLS)}')


if __name__ == '__main__':
    print('Crawling started...')
    start_time = time.time()

    if not load_state():
        print('No saved state found, starting fresh crawl.')
        fetch_robots_txt()

    crawl()
    end_time = time.time()  
    print('Total crawling time: {:.2f} seconds'.format(end_time - start_time))
    print('Crawling finished.')
