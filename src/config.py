URL = 'https://inshorts.com/en/ajax/more_news'
N_PAGES = 1 # Number of pages to scrap
DATA_DIR = '../data/'
MODEL_DIR = '../model'
FILE_NAME = 'news_summary.csv'
TEST_SIZE = 0.1
MODEL_CHECKPOINT = 't5-small'
FINETUNED_MODEL_CHECKPOINT = 'radhikabansal/t5-base-finetuned-news-summary'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 60
BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01