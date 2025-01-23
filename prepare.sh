#!/bin/bash
# re-validate login information
export CURRENT_PORT=1
export CLASSIFIEDS="http://localhost:9984"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://127.0.0.1:7770"
export REDDIT="http://127.0.0.1:9999"
export WIKIPEDIA="http://127.0.0.1:8888"
export HOMEPAGE="http://127.0.0.1:4399"
export OPENAI_API_KEY=""
export DATASET="visualwebarena"
export UGROUND_API="http://127.0.0.1:8000"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

rm -rf ./.auth
mkdir -p ./.auth
python browser_env/auto_login.py