# GenX for Wave Energy

0. Clone this repo: `git clone git@github.com:symbiotic-engineering/WEC-DECIDER.git`
1. Install anaconda, miniconda, or mamba and setup the python environment: `conda env create -f environment.yml`.
2. For the powergenome environment, two edits must be made to powergenome. Find the package at `/envs/wec-decider/lib/python3.10/site-packages/powergenome/`. Change line 212 of `run_powergenome_multiple_outputs_cli.py` and line 124 of `util.py`. See [here](https://github.com/PowerGenome/PowerGenome/pulls?q=is%3Apr+author%3Arebeccamccabe) for the exact changes.
3. Install Julia using the instructions [here](https://julialang.org/downloads/). Use julia version 1.8 (recommend setting it as default).
4. Install Gurobi Optimizer using the "full installation" instructions [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer).
5. Obtain a Gurobi license (ie academic single-user) and install it (ie using `grbgetkey`). If the command is not found, use the full filepath, as described [here](https://support.gurobi.com/hc/en-us/articles/360040113232-How-do-I-resolve-the-error-grbgetkey-command-not-found-or-grbgetkey-is-not-recognized).
6. Configure `h5pyd` according to the instructions [here](https://mhkit-software.github.io/MHKiT/WPTO_hindcast_example.html).
7. First run `make_batch_csv.ipynb` which should generate `replacements.csv`.
8. Then run `caserunner.jl` to run the optimization. To do this, use the following julia commands: 
```
julia # enter julia REPL
using Pkg # load package manager
Pkg.activate("modules/CEM") # set active project - you should be in the WEC-DECIDER directory when executing this command
Pkg.instantiate() # first time only. If on windows, see special instructions in GenX readme.
] # enter package manager
st # status - check that packages installed correctly
backspace or ctrl-C  # exit package manager
Pkg.build("Gurobi") # may be necessary for first Gurobi use?
cd("modules/CEM")
include("caserunner.jl")
```
9. Finally run `analyze_results.ipynb` to generate plots.

See the report `report.pdf` for details on the methodology and results.

Authors: @rebeccamccabe and @AlannLiu