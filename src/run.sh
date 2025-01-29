#!/bin/bash

source .env
fx_estimate_path=${PROJECT_PATH}fx_estimate/

exp_num=$1
source ${fx_estimate_path}src/.venv/bin/activate
(time python3 ${fx_estimate_path}src/main.py ${exp_num}) > run.log 2>&1
deactivate
