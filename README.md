# claudegraph

## ***visualise your ~./claude/history.jsonl conversations***

<img width="1824" height="996" alt="demo" src="https://github.com/user-attachments/assets/78155896-f19e-4535-ad3f-a6b18e0790fc" />

## Build

```sh
git submodule update --init --recursive

cmake -S . -B build
cmake --build build -j
./build/convograph
```


## Structure

Nodes = Conversations

Edges = Similarity

Each conversation becomes a bag of words (title + prompts, lowercased,
stop-worded, tokenized). Those are turned into TF-IDF vectors and compared
with cosine similarity. conversations in the same project also get a boost. Each node keeps its strongest few links unioned into an undirected edge set.
