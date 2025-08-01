#!/bin/bash
set -e # if any command fails, exit immediately. Comment this out if you are using a debugger and have breakpoints on errors.

function run_notebook() {
    local folder="$1"
    local notebook_path="$2"
    jupyter nbconvert --execute "$folder/$notebook_path" --to notebook --output "$notebook_path"
}

echo "$(date +"%Y-%m-%d %H:%M:%S") Activating conda environment"
# to avoid conda error, run this script as "bash -i modules/CEM/run_all.sh"
conda init
conda activate wec-decider-decider-2 # a clone of wec-decider-7-pg-edit but with calkit/nbconvert installed

echo "$(date +"%Y-%m-%d %H:%M:%S") Creating sweep inputs"
run_notebook modules/CEM make_additional_tech_csv.ipynb

echo "$(date +"%Y-%m-%d %H:%M:%S") Downloading PowerGenome data"
bash modules/CEM/get-powergenome-data.sh
echo "$(date +"%Y-%m-%d %H:%M:%S") Running PowerGenome"
bash modules/CEM/run_powergenome.sh

echo "$(date +"%Y-%m-%d %H:%M:%S") Running GenX"
julia modules/CEM/setup.jl
julia modules/CEM/Run.jl

echo "$(date +"%Y-%m-%d %H:%M:%S") Plotting results"
run_notebook modules/CEM analyze_results.ipynb