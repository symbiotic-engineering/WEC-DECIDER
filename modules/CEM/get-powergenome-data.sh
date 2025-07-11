# this script must be run with working directory set to WEC-DECIDER/modules/CEM folder

# download zip files and unzip them
# once powergenome data sources are hosted on zenodo instead of drive, use pooch for this instead of gdown
if ! [ -d ./data/cambium ]; then
    gdown --quiet -O data/cambium.zip 1nbhWwOsNeOtcUew9Mn4QGuAtCsZo0VZ2
    unzip -n ./data/cambium.zip -d ./data/cambium
fi

if ! [ -d ./data/efs ]; then
    gdown --quiet -O data/efs.zip 1dWA35bQpPksnSb6auybMbrIqyaBG6wBM
    unzip -n ./data/efs.zip -d ./data/efs
fi

if ! [ -d ./data/pg ]; then
    gdown --quiet -O data/pg.zip 1AT7vsfxLsKuf9N2JXBTlrt2-4I8Rg_hI
    unzip -n ./data/pg.zip -d ./data/pg
fi
wget --no-verbose -nc -P data/pg "https://github.com/PowerGenome/PowerGenome/raw/refs/heads/master/tests/data/_pg_misc_tables.sqlite3"

if ! [ -d ./data/pudl ]; then
    gdown --quiet -O data/pudl.zip 1tJipxJYxP_dcAnopJrdXdcZh7K3SlI1-
    unzip -n ./data/pudl.zip -d ./data/pudl
fi

# delete zips after extraction
rm -f ./data/*.zip

# these ones are folders, not zips
if ! [ -d ./data/resource_profiles ]; then
    gdown --quiet --folder -O data/resource_profiles 1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG 
fi

if ! [ -d ./data/resource_groups ]; then
    gdown --quiet --folder -O data/resource_groups 1Svkz6fKgc1m9ewUMPjVHJJV5TWDYKdmw
fi

if ! [ -d ./data/network_costs ]; then
    gdown --quiet --folder -O data/network_costs 16bnl3VSUMP8UNEhA881VGpFqCkmeadcm
fi

if ! [ -d ./data/extra_inputs ]; then
    gdown --quiet --folder -O data/extra_inputs 1dQt1Drk8wkWU-T3BO8zUlg4yUf1euYJx
fi

# files from powergenome
url1="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/misc_gen_inputs.csv"
url2="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/demand_segments_voll.csv"
url3="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/resource_capacity_spur.csv"
url4="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/emission_policies.csv"
url5="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/Reserves.csv"
wget -nc -P data_east $url1 $url2 $url3 $url4 $url5

url6="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_demand_segments_voll.csv"
url7="https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_misc_gen_inputs.csv"
wget -nc -P data_CA $url6 $url7
