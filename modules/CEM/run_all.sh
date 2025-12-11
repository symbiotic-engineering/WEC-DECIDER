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
# conda activate wec-decider-backup # a clone of wec-decider-7-pg-edit but with calkit/nbconvert installed
source /usr/share/miniconda/etc/profile.d/conda.sh
conda activate wec-decider-decider-2 # a clone of wec-decider-7-pg-edit but with calkit/nbconvert installed

echo "$(date +"%Y-%m-%d %H:%M:%S") Creating sweep inputs"
run_notebook modules/CEM make_additional_tech_csv.ipynb

echo "$(date +"%Y-%m-%d %H:%M:%S") Downloading PowerGenome data"
bash modules/CEM/get-powergenome-data.sh

echo "$(date +"%Y-%m-%d %H:%M:%S") Running PowerGenome"
bash modules/CEM/run_powergenome.sh

echo "$(date +"%Y-%m-%d %H:%M:%S") Running GenX"
julia modules/CEM/setup.jl
julia modules/CEM/Run.jl "Case_year_2030_electrification_ref_carbon_constraint_med_wave_variability_Avail_zeta_0.05_omega_n_0.5_D_f_20_limited_100_wave_cost_Wave_400"

echo "$(date +"%Y-%m-%d %H:%M:%S") Plotting results"
run_notebook modules/CEM analyze_results.ipynb
