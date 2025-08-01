set -e # if any command fails, exit immediately. Comment this out if you are using a debugger and have breakpoints on errors.

# this script must be run with working directory set to WEC-DECIDER/modules/CEM folder
script_dir=$(dirname -- "$0")

if ! [ -d $script_dir/data ]; then
    mkdir -p $script_dir/data
fi

# download zip files and unzip them
# once powergenome data sources are hosted on zenodo instead of drive, use pooch for this instead of gdown
if ! [ -d $script_dir/data/cambium ]; then
    gdown --quiet -O $script_dir/data/cambium.zip 1nbhWwOsNeOtcUew9Mn4QGuAtCsZo0VZ2
    unzip -n $script_dir/data/cambium.zip -d $script_dir/data/cambium
fi

if ! [ -d $script_dir/data/efs ]; then
    gdown --quiet -O $script_dir/data/efs.zip 1dWA35bQpPksnSb6auybMbrIqyaBG6wBM
    unzip -n $script_dir/data/efs.zip -d $script_dir/data/efs
fi

if ! [ -d $script_dir/data/pg ]; then
    gdown --quiet -O $script_dir/data/pg.zip 1AT7vsfxLsKuf9N2JXBTlrt2-4I8Rg_hI
    unzip -n $script_dir/data/pg.zip -d $script_dir/data/pg
fi
wget --no-verbose -nc -P $script_dir/data/pg "https://github.com/PowerGenome/PowerGenome/raw/refs/heads/master/tests/data/_pg_misc_tables.sqlite3"

if ! [ -d $script_dir/data/pudl ]; then
    gdown --quiet -O $script_dir/data/pudl.zip 1tJipxJYxP_dcAnopJrdXdcZh7K3SlI1-
    unzip -n $script_dir/data/pudl.zip -d $script_dir/data/pudl
fi

# delete zips after extraction
rm -f $script_dir/data/*.zip

# these ones are folders, not zips
if ! [ -d $script_dir/data/resource_profiles ]; then
    gdown --quiet --folder -O $script_dir/data/resource_profiles 1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG 
fi

if ! [ -d $script_dir/data/resource_groups ]; then
    gdown --quiet --folder -O $script_dir/data/resource_groups 1Svkz6fKgc1m9ewUMPjVHJJV5TWDYKdmw
fi

if ! [ -d $script_dir/data/network_costs ]; then
    gdown --quiet --folder -O $script_dir/data/network_costs 16bnl3VSUMP8UNEhA881VGpFqCkmeadcm
fi

if ! [ -d $script_dir/data/extra_inputs ]; then
    gdown --quiet --folder -O $script_dir/data/extra_inputs 1dQt1Drk8wkWU-T3BO8zUlg4yUf1euYJx
fi

# files from powergenome
url1="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/misc_gen_inputs.csv"
url2="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/demand_segments_voll.csv"
url3="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/resource_capacity_spur.csv"
url4="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/emission_policies.csv"
url5="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/Reserves.csv"
wget --no-verbose -nc --directory-prefix $script_dir/data_east $url1 $url2 $url3 $url4 $url5
# -nc means if file already exists, do not download it again

url6="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_demand_segments_voll.csv"
url7="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_misc_gen_inputs.csv"
wget --no-verbose -nc --directory-prefix $script_dir/data_CA $url6 $url7
