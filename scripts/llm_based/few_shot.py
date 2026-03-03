import pandas as pd
from pandas import Series
from rank_bm25 import BM25Okapi

class BM25Selector:
    def __init__(self, examples: Series):
        tokenized_examples = [self._tokenizer(example) for example in examples]
        self.model = BM25Okapi(tokenized_examples)

    
    def _tokenizer(text):
        return text.split()
    
    def get_model(self):

    def get_examples()

    



  
if __name__ == "__main__":
    data = pd.read_json("data/sample_data.jsonl", lines=True)
    bm25 = BM25Selector(data["code"])
    