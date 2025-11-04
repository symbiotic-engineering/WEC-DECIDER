# Capacity Expansion Modeling (CEM) for Wave Energy

Instructions for use:

0. Clone this repo: `git clone git@github.com:symbiotic-engineering/WEC-DECIDER.git`
1. Install anaconda, miniconda, or mamba and setup the python environment: `conda env create -f envs/calkit-environment.yml`.
2. Install Julia using the instructions [here](https://julialang.org/downloads/). Use julia version 1.8 or 1.9 (recommend setting it as default).
3. Install Gurobi Optimizer using the "full installation" instructions [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer).
4. Obtain a Gurobi license (ie academic single-user) and install it (ie using `grbgetkey`). If the command is not found, use the full filepath, as described [here](https://support.gurobi.com/hc/en-us/articles/360040113232-How-do-I-resolve-the-error-grbgetkey-command-not-found-or-grbgetkey-is-not-recognized).
5. Configure `h5pyd` according to the instructions [here](https://mhkit-software.github.io/MHKiT/WPTO_hindcast_example.html):
  ```
  conda activate wec-decider-decider-2
  hsconfigure
  (now paste your nrel code in here when prompted)
  ```
6. `cd` to the `WEC-DECIDER` folder and run `bash -i modules/CEM/run_all.sh`. This defines the sweep, downloads PowerGenome data inputs, runs PowerGenome, runs GenX, and plots results. (Eventually this step will be replaced by `calkit run`).
If PowerGenome gives an error 'No resource groups specified {paths}', that probably means your download of power genome data failed. This can be solved by manually downloading the relvant files from google drive: https://drive.google.com/drive/folders/1K5GWF5lbe-mKSTUSuJxnFdYGCdyDJ7iE (this link is from the power genome readme). You will need to place them in the correct subfolder in the data folder.
Also note that some of the powergenome google drive resource groups have metadata that doesn't match the filenames. They will need to be manually fixed by adding `_lcoe` to the metadata jsons - see [here](https://github.com/PowerGenome/PowerGenome/issues/395).

8. If making any updates, re-run individual parts of the process as needed using the commands in `run_all.sh`. If you want to run GenX from the interactive Julia REPL, use the following commands: 
```
julia # enter julia REPL
using Pkg # load package manager
Pkg.activate("modules/CEM") # set active project - you should be in the WEC-DECIDER directory when executing this command
Pkg.instantiate() # first time only. If on windows, see special instructions in GenX readme.
] # enter package manager
st # status - check that packages installed correctly
backspace or ctrl-C  # exit package manager
Pkg.build("Gurobi") # first time only
cd("modules/CEM")
include("Run.jl")
```

For details on the methodology and results, see the report `old/report.pdf` for early work, and the conference paper [here](https://github.com/symbiotic-engineering/MDOcean/tree/decider-pub/pubs/UMERC-2025-grid-value) for more recent work.

Code authors: @rebeccamccabe, @Khai003, @AlannLiu
