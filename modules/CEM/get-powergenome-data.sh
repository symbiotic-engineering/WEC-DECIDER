# this script must be run with working directory set to WEC-DECIDER/ folder

# download zip files and unzip them
# once powergenome data sources are hosted on zenodo instead of drive, use pooch for this instead of gdown
if ! [ -d ./data/cambium ]; then
    gdown -O data/cambium.zip 1nbhWwOsNeOtcUew9Mn4QGuAtCsZo0VZ2
    unzip -n ./data/cambium.zip -d ./data/cambium
fi

if ! [ -d ./data/efs ]; then
    gdown -O data/efs.zip 1dWA35bQpPksnSb6auybMbrIqyaBG6wBM
    unzip -n ./data/efs.zip -d ./data/efs
fi

if ! [ -d ./data/pg ]; then
    gdown -O data/pg.zip 1XrLOqVGNP1qjvsXeTt1YH2Pyppqad0fc
    unzip -n ./data/pg.zip -d ./data/pg
fi

if ! [ -d ./data/pudl ]; then
    gdown -O data/pudl.zip 1tJipxJYxP_dcAnopJrdXdcZh7K3SlI1-
    unzip -n ./data/pudl.zip -d ./data/pudl
fi

# delete zips after extraction
rm -f ./data/*.zip

if ! [ -d ./data/resource_profiles ]; then
    gdown --folder -O data/resource_profiles 1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG # this one isn't a zip
fi

# run sweep
run_powergenome_multiple -sf ./modules/CEM/sweeps_settings.yml -rf ./modules/CEM/results