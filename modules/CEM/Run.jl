using Gurobi
using GenX
using Pkg

# 1. Determine the directory where this script lives
script_dir = @__DIR__
cem_dir = joinpath(script_dir, "..", "..")

# 2. Activate and instantiate the environment at modules/CEM
Pkg.activate(cem_dir)
Pkg.instantiate()
Pkg.build("Gurobi")


cd(cem_dir)

# Include and run the case runner logic
# uncomment to run caserunner from 3rd party
# include(joinpath(cem_dir, "caserunner_v1.jl"))

# Uncomment to run caserunner from original GenX GitHub version
run_genx_case!(dirname(@__FILE__), Gurobi.Optimizer)
