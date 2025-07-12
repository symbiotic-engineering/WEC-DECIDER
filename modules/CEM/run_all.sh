#!/bin/bash
set -e # if any command fails, exit immediately. Comment this out if you are using a debugger and have breakpoints on errors.

function run_notebook() {
    local folder="$1"
    local notebook_path="$2"
    jupyter nbconvert --execute "$folder/$notebook_path" --to notebook --output "$notebook_path"
}

# to avoid conda error, run this script as "bash -i modules/CEM/run_all.sh"
conda activate wec-decider-decider-2 # a clone of wec-decider-7-pg-edit but with calkit/nbconvert installed

run_notebook modules/CEM make_additional_tech_csv.ipynb

bash modules/CEM/get-powergenome-data.sh
bash modules/CEM/run_powergenome.sh

#julia modules/CEM/setup.jl
julia modules/CEM/Run.jl

run_notebook modules/CEM analyze_results.ipynb