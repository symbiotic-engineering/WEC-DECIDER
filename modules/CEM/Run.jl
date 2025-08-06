using Pkg

# Activate the project environment
cem_dir = @__DIR__
Pkg.activate(cem_dir)

cd(cem_dir)

using Gurobi
using GenX
using Infiltrator
using YAML

# Include and run the case runner logic
# uncomment to run caserunner from 3rd party
#include(joinpath(cem_dir, "caserunner.jl"))

# Uncomment to run caserunner from original GenX GitHub version
function run_debug()
    case_folder_dir = joinpath(cem_dir, "data_east", "cases")
    num_cases = sum(occursin.("Case_", readdir(case_folder_dir)))
    for i=1:num_cases
        if occursin("Case_", readdir(case_folder_dir)[i])
            println("Running GenX for case: ", readdir(case_folder_dir)[i], " of ", num_cases)
            case_dir = joinpath(case_folder_dir, readdir(case_folder_dir)[i])

            # contents of run_genx_case!(case_dir, Gurobi.Optimizer)
            genx_settings = GenX.get_settings_path(case_dir, "genx_settings.yml") # Settings YAML file path
            writeoutput_settings = GenX.get_settings_path(case_dir, "output_settings.yml") # Write-output settings YAML file path
            mysetup = GenX.configure_settings(genx_settings, writeoutput_settings)

            # contents of run_genx_case_multistage!
            settings_path = GenX.get_settings_path(case_dir)
            multistage_settings = GenX.get_settings_path(case_dir, "multi_stage_settings.yml") # Multi stage settings YAML file path
            
            # contents of configure_settings_multistage(multistage_settings)
            # merge default settings with those specified in the YAML file
            model_settings = isfile(multistage_settings) ? YAML.load(open(multistage_settings)) : Dict{Any, Any}()
            settings = GenX.default_settings_multistage()
            merge!(settings, model_settings)
            GenX.validate_multistage_settings!(settings)
            # end configure_settings_multistage

            mysetup["MultiStageSettingsDict"] = settings
            num = mysetup["MultiStageSettingsDict"]["NumStages"]
            println("Number of Stages: ", num)
            @infiltrate # creates a breakpoint for inspecting settings

            run_genx_case_multistage!(case_dir, mysetup, Gurobi.Optimizer)
        end
    end
end

function run()
    force = false # true to re-run cases that have already been run

    case_folder_dir = joinpath(cem_dir, "data_east", "cases")
    num_cases = sum(occursin.("Case_", readdir(case_folder_dir)))
    for i=1:num_cases
        if occursin("Case_", readdir(case_folder_dir)[i])
            case_dir = joinpath(case_folder_dir, readdir(case_folder_dir)[i])
            already_run = isdir(joinpath(case_dir, "results"))
            if already_run && !force
                println("Skipping GenX case ", i, " of ", num_cases, " (already run)")
            else
                println("Running GenX for case ", i, " of ", num_cases)
                run_genx_case!(case_dir, Gurobi.Optimizer)
            end
        end
    end
end

# run_debug()
run()
