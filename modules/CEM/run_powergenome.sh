#!/bin/bash
set -e # if any command fails, exit immediately. Comment this out if you are using a debugger and have breakpoints on errors.

script_dir=$(cd "$(dirname "$0")" && pwd) # absolute path to this script

function run_for_one_location {
    echo "======================================================================="
    local location=$1

    # Copy env variables to settings folder and expand them
    envsubst < "$script_dir/pg-data-env.yml" > "$script_dir/$location/settings/pg-data-env-expanded.yml"

    run_powergenome_multiple -sf "$script_dir/$location/settings/" -rf "$script_dir/$location/cases"
    for i in "$script_dir/$location/cases"/Case_*; do
        cp -r "$script_dir/template/settings/" "$i"
    done

    echo "PowerGenome ran for $location."
}

export PG_DATA_FOLDER="${script_dir}/data"
echo "Using data folder: $PG_DATA_FOLDER"

run_for_one_location "data_east"
#run_for_one_location "data_CA"

