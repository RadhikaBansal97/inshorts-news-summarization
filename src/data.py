import pandas as pd
import numpy as np
from newspaper import Article
from tqdm import tqdm

def getArticleText(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    return text

if __name__=="__main__":
    df = pd.read_csv('../data/inshorts_data.csv')
    for i, url in tqdm(enumerate(df['readMoreUrl'])):
        df.loc[i, 'article'] = getArticleText(url)
    df =df[['article', 'content','readMoreUrl']]
    df.to_csv('../data/data.csv',index=False)
