from datasets import load_dataset
import config

if __name__=="__main__":
    news_datasets = load_dataset('csv', data_files=config.FILE_NAME,split ='train', data_dir = config.DATA_DIR)
    train_test = news_datasets.train_test_split(config.TEST_SIZE, seed=42)
    train_val_test = train_test["train"].train_test_split(config.TEST_SIZE, seed = 42)

    train_val_test['validation'] = train_val_test['test']
    train_val_test['test'] = train_test.pop('test')
    news_dataset = train_val_test

    if config.MODEL_CHECKPOINT.split('-')[0]=='t5':
        news_dataset = news_dataset.map(lambda x : {'summary':'summarize: '+x['summary']})

    news_dataset.save_to_disk(config.DATA_DIR)