import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from tqdm import tqdm
import config 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getInshortsNews(response_text):

    newsData = []
    soup = BeautifulSoup(response_text, 'lxml')
    newsCards = soup.find_all(class_='news-card')

    for card in newsCards:
        try:
            title = card.find(class_='news-card-title').find('a').text
        except AttributeError:
            title = None

        try:
            url = ('https://www.inshorts.com' + card.find(class_='news-card-title')
                   .find('a').get('href'))
        except AttributeError:
            url = None

        try:
            content = card.find(class_='news-card-content').find('div').text
        except AttributeError:
            content = None

        try:
            author = card.find(class_='author').text
        except AttributeError:
            author = None

        try:
            time = card.find(class_='time')['content']
        except AttributeError:
            time = None
            
        try:
            source = card.find(class_="news-card-footer news-right-box").find('a').get_text()
        except AttributeError:
            source = None
        try:
            readMoreUrl = card.find(class_='read-more').find('a').get('href')
        except AttributeError:
            readMoreUrl = None

        newsObject = {
            'title': title,
            'url': url,
            'content': content,
            'author': author,
            'time': time,
            'source':source,
            'readMoreUrl': readMoreUrl
        }
        newsData.append(newsObject)

    return newsData

if __name__=="__main__":

    # load first page:
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    html_doc = requests.get('https://www.inshorts.com/en/read', headers=agent).text
    min_news_id = re.search(r'min_news_id = "([^"]+)"', html_doc).group(1)

    logger.info(f"Successfully extracted the offest of first page - {min_news_id}")
    url = config.url
    news_offset = min_news_id
    n_pages = config.n_page
    news_data = []
    logger.info(f"Start scrapping the website for {n_pages} pages")
    for i in tqdm(range(n_pages)):
        while True:
            response = requests.post(url, data={"category": "", "news_offset": news_offset})
            if response.status_code == 200:
                break
            else:
                print(response.status_code)

        response_json = json.loads(response.text)
        news_article= getInshortsNews(response_json["html"]),
        news_offset = response_json["min_news_id"]
        news_data.extend(news_article[0])

    df = pd.DataFrame(news_data)
    df = df[df['source'].isin(['Hindustan Times','Reuters','Times Now','The Print','Free Press Journal','News18','Sportskeeda','Moneycontrol','Press Trust of India'])]
    df = df[df["readMoreUrl"].str.contains("twitter.com|youtube.com|youtu.be") == False].reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    df.to_csv('../data/inshorts_data.csv', index=False)
    logger.info("Successfully scrapped data from inshorts")