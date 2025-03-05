# Pipeline Demo (EDF and Provenance Store)

## Overview
This project is divided into the following sections:
- Error Detection (`edf.py`) and categorization (`error.py`)
- Pipeline definition, stages, and pipeline generation (`pipeline.py`)
- Agent library (`agent.py`)
- Provenance Store (TODO: not implemented)
- Example pipelines (`examples/`)
    - `grocery.py` is a simple pipeline that loads data, cleans it, and merges it with web search results, as well as detecting errors for a self healing pipeline

## Installation
Start Redis server to cahce intermediate result
```bash
brew install redis
redis-server
```

After setting up the project as with an average `uv` project, run example pipeline:
```bash
cd examples/data && uv run kaggle datasets download -d polartech/walmart-grocery-product-dataset && unzip walmart-grocery-product-dataset.zip

uv run python -m examples.grocery
```


### TODO
- add actual lineage (add lineage to data, not just edfs across pipelines)
- make api endpoints flaky to see network errors (hard)
- using error registrations for healing / improve future runs
- introduce chaining with https://github.com/microsoft/autogen

feedback
- visualization of stages and expose info about sources at each stage to optimizer
- hard problem: knowing when model is right
- some errors may be correlated
- different sources have trust