import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import src.config as config
from newspaper import Article

model_checkpoint = f"radhikabansal/mt5-small-finetuned-amazon-en-es"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

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
    
    return pred[0][10:]

def summary(url):
    try:
        article = fetch_newsarticle(url)
        return model_pred(article)
    except:
        return 'Oops! sorry i am not trained on this source URL'

def main():   
    # We instantiate the Textbox class
    textbox = gr.Textbox(label="Place the news URL here:", lines=2)

    demo = gr.Interface(fn=summary, inputs=textbox, outputs="text",
                allow_flagging="manual",
                title="News Summarizer Demo",
                examples=[
                ["https://www.reuters.com/world/france-accuses-russia-stoking-armenia-azerbaijan-conflict-2022-10-12/"],
                ["https://www.hindustantimes.com/cities/delhi-news/delhi-l-g-office-says-capital-s-landfill-sites-went-down-by-462-in-just-4-mths-101665588501589.html"],
                ["https://www.moneycontrol.com/news/business/companies/edtech-firm-veranda-learning-solutions-to-acquire-jk-shah-education-9320841.html"],
        ],
        cache_examples=True)
    demo.launch(server_name="0.0.0.0", server_port=7000, share=True)

if __name__ == "__main__":
    main()