# claudegraph

## ***visualise your ~./claude/history.jsonl conversations***

## Build

```sh
git submodule update --init --recursive

cmake -S . -B build
cmake --build build -j
./build/convograph
```

## How "related" is decided

Each conversation becomes a bag of words (title + prompts, lowercased,
stop-worded, tokenized). Those are turned into TF-IDF vectors and compared
with cosine similarity. conversations in the same project also get a boost. Each node keeps its strongest few links unioned into an undirected edge set.