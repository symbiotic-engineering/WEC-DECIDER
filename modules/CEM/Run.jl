using Pkg

# 1. Determine the directory where this script lives
script_dir = @__DIR__
cem_dir = joinpath(script_dir)

# 2. Activate and instantiate the environment at modules/CEM
Pkg.activate(cem_dir)
Pkg.instantiate()
Pkg.build("Gurobi")

cd(cem_dir)

using Gurobi
using GenX

# Include and run the case runner logic
# uncomment to run caserunner from 3rd party
#include(joinpath(cem_dir, "caserunner.jl"))

# Uncomment to run caserunner from original GenX GitHub version
case_folder_dir = joinpath(cem_dir, "data_east", "results")
for i=1:length(readdir(case_folder_dir))
    if occursin("Case_", readdir(case_folder_dir)[i])
        println("Running GenX for case: ", readdir(case_folder_dir)[i], " of ", length(readdir(case_folder_dir)))
        case_dir = joinpath(case_folder_dir, readdir(case_folder_dir)[i],"Inputs","Inputs_p1")
        run_genx_case!(case_dir, Gurobi.Optimizer)
    end
end

