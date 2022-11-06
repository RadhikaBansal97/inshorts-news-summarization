from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import config
from newspaper import Article

url = input("Please enter the article URL you want to summarize: ")
print("You entered: " + str(url))

def fetch_newsarticle(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def model_pred(article):
    
    inputs = tokenizer(article, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(input_ids, attention_mask=attention_mask, min_length = 40, max_length = 80)

    # all special tokens including will be removed
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return pred[0]

if __name__=="__main__":
    model_checkpoint = f"{config.MODEL_DIR}/{config.FINETUNED_MODEL_CHECKPOINT}"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    article = fetch_newsarticle(url)
    news_summary = model_pred(article)

    if model_checkpoint[:2] == 't5':
        print(news_summary[10:])
    else:
        print(news_summary)