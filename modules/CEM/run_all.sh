#!/bin/bash

conda init
conda activate wec-decider-decider # a clone of wec-decider-7-pg-edit but with calkit/nbconvert installed

run_notebook("modules/CEM/make_additional_tech_csv.ipynb")

bash modules/CEM/get-powergenome-data.sh
bash modules/CEM/run_powergenome.sh

#julia modules/CEM/setup.jl
julia modules/CEM/Run.jl

run_notebook("modules/CEM/analyze_results.ipynb")

function run_notebook() {
    local notebook_path="$1"
    jupyter nbconvert --execute "$notebook_path" --to notebook --output "$notebook_path"
}