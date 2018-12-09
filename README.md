# tfidf
basic tfidf counting

## usage
```python
# create object
tfidf = TFIDF()

# update with stuff (read from files)
for doc_path in doc_paths:
    with io.open(doc_path, mode='r', encoding='utf8') as f:
        tfidf.update(f.read().split(), doc_path)
    
# get TF-IDF for top-10 terms
for doc_id, word_bm25 in tfidf.generate_bm25().iteritems():
    print(doc_id, word_bm25.most_common(10))


```