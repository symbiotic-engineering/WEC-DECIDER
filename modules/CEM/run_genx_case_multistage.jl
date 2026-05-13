module RunGenXCaseMultiStage

using JuMP
using CSV, DataFrames
using GenX:
    get_settings_path,
    configure_settings,
    prevent_doubled_timedomainreduction,
    time_domain_reduced_files_exist,
    cluster_inputs,
    configure_solver,
    load_inputs,
    generate_model,
    solve_model,
    get_default_output_folder,
    write_outputs,
    mga,
    morris,
    has_values,
    configure_settings_multistage,
    configure_multi_stage_inputs,
    compute_cumulative_min_retirements!,
    validate_can_retire_multistage,
    choose_output_dir,
    run_myopic_multistage,
    run_ddp,
    write_multi_stage_outputs

export run_genx_case_multistage!


function run_genx_case_multistage!(case::AbstractString, mysetup::Dict, optimizer::Any)
    settings_path = get_settings_path(case)
    multistage_settings = get_settings_path(case, "multi_stage_settings.yml") # Multi stage settings YAML file path
    # merge default settings with those specified in the YAML file
    mysetup["MultiStageSettingsDict"] = configure_settings_multistage(multistage_settings)

    ### Cluster time series inputs if necessary and if specified by the user
    if mysetup["TimeDomainReduction"] == 1
        tdr_settings = get_settings_path(case, "time_domain_reduction_settings.yml") # Multi stage settings YAML file path
        TDRSettingsDict = YAML.load(open(tdr_settings))

        first_stage_path = joinpath(case, "inputs", "inputs_p1")
        TDRpath = joinpath(first_stage_path, mysetup["TimeDomainReductionFolder"])
        system_path = joinpath(first_stage_path, mysetup["SystemFolder"])
        prevent_doubled_timedomainreduction(system_path)
        if !time_domain_reduced_files_exist(TDRpath)
            if (mysetup["MultiStage"] == 1) &&
               (TDRSettingsDict["MultiStageConcatenate"] == 0)
                println("Clustering Time Series Data (Individually)...")
                for stage_id in 1:mysetup["MultiStageSettingsDict"]["NumStages"]
                    cluster_inputs(case, settings_path, mysetup, stage_id)
                end
            else
                println("Clustering Time Series Data (Grouped)...")
                cluster_inputs(case, settings_path, mysetup)
            end
        else
            println("Time Series Data Already Clustered.")
        end
    end

    ### Configure solver
    println("Configuring Solver")
    solver_name = lowercase(get(mysetup, "Solver", ""))
    OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

    model_dict = Dict()
    inputs_dict = Dict()

    for t in 1:mysetup["MultiStageSettingsDict"]["NumStages"]

        # Step 0) Set Model Year
        mysetup["MultiStageSettingsDict"]["CurStage"] = t

        # Step 1) Load Inputs
        inpath_sub = joinpath(case, "inputs", string("inputs_p", t))

        inputs_dict[t] = load_inputs(mysetup, inpath_sub)
        inputs_dict[t] = configure_multi_stage_inputs(inputs_dict[t],
            mysetup["MultiStageSettingsDict"],
            mysetup["NetworkExpansion"])

        compute_cumulative_min_retirements!(inputs_dict, t)
        # Step 2) Generate model
        model_dict[t] = generate_model(mysetup, inputs_dict[t], OPTIMIZER)
    end

    # check that resources do not switch from can_retire = 0 to can_retire = 1 between stages
    validate_can_retire_multistage(
        inputs_dict, mysetup["MultiStageSettingsDict"]["NumStages"])

    # Prepare folder for results    
    outpath = get_default_output_folder(case)

    if mysetup["OverwriteResults"] == 1
        # Overwrite existing results if dir exists
        # This is the default behaviour when there is no flag, to avoid breaking existing code
        if !(isdir(outpath))
            mkdir(outpath)
        end
    else
        # Find closest unused ouput directory name and create it
        outpath = choose_output_dir(outpath)
        mkdir(outpath)
    end

    ### Solve model
    println("Solving Model")

    # Step 3) Run DDP Algorithm or Myopic single pass
    if mysetup["MultiStageSettingsDict"]["Myopic"] == 1
        mystats_d = Dict()  # mystats_d is for DDP iteration metadata
        model_dict, inputs_dict = run_myopic_multistage(outpath, model_dict, mysetup, inputs_dict)
    else
        model_dict, mystats_d, inputs_dict = run_ddp(outpath, model_dict, mysetup, inputs_dict)
    end

    # Step 4) Write final outputs from each stage
    if mysetup["MultiStageSettingsDict"]["Myopic"] == 0 ||
       mysetup["MultiStageSettingsDict"]["WriteIntermittentOutputs"] == 0
        for p in 1:mysetup["MultiStageSettingsDict"]["NumStages"]
            mysetup["MultiStageSettingsDict"]["CurStage"] = p
            outpath_cur = joinpath(outpath, "results_p$p")
            write_outputs(model_dict[p], outpath_cur, mysetup, inputs_dict[p])
        end
    end

    # Step 5) Write DDP summary outputs

    write_multi_stage_outputs(mystats_d, outpath, mysetup, inputs_dict)
end

end #module