module RunGenXCaseCustom

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
    has_values

export run_genx_case_simple!

# function run_genx_case_simple!(case::AbstractString, mysetup::Dict, optimizer::Any)
#     settings_path = get_settings_path(case)

#     ### Cluster time series inputs if necessary and if specified by the user
#     if mysetup["TimeDomainReduction"] == 1
#         TDRpath = joinpath(case, mysetup["TimeDomainReductionFolder"])
#         system_path = joinpath(case, mysetup["SystemFolder"])
#         prevent_doubled_timedomainreduction(system_path)
#         if !time_domain_reduced_files_exist(TDRpath)
#             println("Clustering Time Series Data (Grouped)...")
#             cluster_inputs(case, settings_path, mysetup)
#         else
#             println("Time Series Data Already Clustered.")
#         end
#     end

#     ### Configure solver
#     println("Configuring Solver")
#     solver_name = lowercase(get(mysetup, "Solver", ""))
#     OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

#     #### Running a case

#     ### Load inputs
#     println("Loading Inputs")
#     myinputs = load_inputs(mysetup, case)

#     println("Generating the Optimization Model")
#     time_elapsed = @elapsed EP = generate_model(mysetup, myinputs, OPTIMIZER)
#     println("Time elapsed for model building is")
#     println(time_elapsed)

#     println("Solving Model")
#     EP, solve_time = solve_model(EP, mysetup)
#     myinputs["solve_time"] = solve_time # Store the model solve time in myinputs

#     # Run MGA if the MGA flag is set to 1 else only save the least cost solution
#     if has_values(EP)
#         println("Writing Output")
#         outputs_path = get_default_output_folder(case)
#         elapsed_time = @elapsed outputs_path = write_outputs(EP,
#             outputs_path,
#             mysetup,
#             myinputs)
#         println("Time elapsed for writing is")
#         println(elapsed_time)
#         if mysetup["ModelingToGenerateAlternatives"] == 1
#             println("Starting Model to Generate Alternatives (MGA) Iterations")
#             mga(EP, case, mysetup, myinputs)
#         end

#         if mysetup["MethodofMorris"] == 1
#             println("Starting Global sensitivity analysis with Method of Morris")
#             morris(EP, case, mysetup, myinputs, outputs_path, OPTIMIZER)
#         end
#     end
# end

# Helper function to safely extract a dual value
function get_dual(con, g)
    if con === nothing
        return NaN
    end

    # DenseAxisArray indexing (the common GenX form)
    inds = con.axes[1]

    if g in inds
        return dual(con[g])
    else
        return NaN
    end
end


function run_genx_case_simple!(case::AbstractString, mysetup::Dict, optimizer::Any)
    println("Case directory = $case")

    settings_path = get_settings_path(case)
    settings_file = joinpath(settings_path, "genx_settings.yml")
    println("GenX settings_file = $settings_file")

    system_path = joinpath(case, "system")
    println("Processed settings will be written to: $system_path")

    # Run GenX settings processor
    settings = configure_settings(settings_file, system_path)

    # Merge settings into mysetup
    for (k, v) in settings
        mysetup[string(k)] = v
    end

    # Add missing defaults to mysetup
    defaults = Dict(
        "OPEXMULT" => 1.0,
        "INV_OP_RESERVE_REQ" => 0,
        "FLEXRAMPING" => 0,
        "CAPACITY_FACTOR" => 1.0,
        "EmissionLimit" => 0.0,
        "WECCF" => 1.0,
        "MultiStageSettingsDict" => Dict{String, Any}(),
        "NumStages" => 1
    )
    for (k, v) in defaults
        k_str = string(k)
        if !haskey(mysetup, k_str)
            println("Setting default $k_str = $v")
            mysetup[k_str] = v
        end
    end

    #This is what gets rid of multistage retirement thingie
    mysetup["MultiStage"] = 0
    mysetup["MultiStageSettingsDict"] = Dict()


    # if !haskey(mysetup, "OPEXMULT")
    #     mysetup["OPEXMULT"] = 1.0
    # end

    ### Cluster time series inputs if necessary
    if mysetup["TimeDomainReduction"] == 1
        TDRpath = joinpath(case, mysetup["TimeDomainReductionFolder"])
        if isdir(joinpath(case, mysetup["SystemFolder"]))
            prevent_doubled_timedomainreduction(joinpath(case, mysetup["SystemFolder"]))
        else
            println("Warning: system folder not found, skipping TDR prevention.")
        end
        if !time_domain_reduced_files_exist(TDRpath)
            println("Clustering Time Series Data (Grouped)...")
            cluster_inputs(case, settings_path, mysetup)
        else
            println("Time Series Data Already Clustered.")
        end
    end

    ### Configure solver
    println("Configuring Solver")
    solver_name = lowercase(get(mysetup, "Solver", ""))
    OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

    ### Load inputs
    println("Loading Inputs")
    inputs_path = joinpath(case, "inputs", "inputs_p1")
    myinputs = load_inputs(mysetup, inputs_path)

    # Add missing defaults to myinputs as well (prevents KeyErrors in GenX modules)
    for (k, v) in defaults
        k_str = string(k)
        if !haskey(myinputs, k_str)
            myinputs[k_str] = v
        end
    end

    ### Generate optimization model
    println("Generating the Optimization Model")
    time_elapsed = @elapsed EP = generate_model(mysetup, myinputs, OPTIMIZER)
    println("Time elapsed for model building: $time_elapsed seconds")

    ### Solve the model
    println("Solving Model")
    EP, solve_time = solve_model(EP, mysetup)
    myinputs["solve_time"] = solve_time

    println(typeof(myinputs["G"]))
    println(typeof(myinputs["RESOURCES"]))

    #print the variables in EP
    open(joinpath(case, "EP_fields.txt"), "w") do io
        println(io, "=== Constraints in EP ===")
        for c in all_constraints(EP; include_variable_in_set_constraints=false)
            cname = JuMP.name(c) === nothing ? "<unnamed>" : JuMP.name(c)
            println(io, cname, " :: ", typeof(c))
        end
    end


    println("EP field names written to EP_fields.txt")

    # #for printing out each variable in EP and their types
    # open(joinpath(case, "EP_field_types.txt"), "w") do io
    #     println(io, "=== Fields and their types ===")
    #     for fn in fieldnames(typeof(EP))
    #         try
    #             val = getfield(EP, fn)
    #             println(io, fn, " => ", typeof(val))
    #         catch
    #             println(io, fn, " => FAILED TO READ")
    #         end
    #     end
    # end

    # println("EP field types written to EP_field_types.txt")


    ### Print all dual constrains in EP object
    output_file = joinpath(case, "EP_duals.txt")
    open(output_file, "w") do io
        println(io, "--- Dual values ---")
        
        # Example: loop through all constraints
        for c in all_constraints(EP; include_variable_in_set_constraints=false)
            # get the constraint name if it exists
            cname = JuMP.name(c)  # gets the constraint's name if it has one
            val = try
                dual(c)
            catch
                missing  # or skip if dual doesn't exist
            end
            println(io, cname, " => ", val)
        end
    end

    println("EP dual constrains written to $output_file")

    println("Writing the marginal thermal duals")
    #Prints the thermal generators max and min power constraints
    
    open(joinpath(case, "thermal_duals_marginal.txt"), "w") do io
        println(io, "gen\ttime\tcMinPowerThermal\tcMaxPowerThermal")

        for t in 1:480
            for g in 4:27
                # Access named constraint references directly
                min_con = EP[:cMinPowerThermal][g,t]
                max_con = EP[:cMaxPowerThermal][g,t]

                # Get dual values
                min_val = try dual(min_con) catch; missing end
                max_val = try dual(max_con) catch; missing end

                if min_val == 0.0 && max_val == 0.0
                    println(io, g, "\t", t, "\t", min_val, "\t", max_val)
                end
            end
        end
    end

    #     println("Extracting power max duals...")
    # powermax = extract_power_max_duals(EP)

    # println("Extracting power min duals...")
    # powermin = extract_power_min_duals(EP)

    # #Convert to DataFrames
    # df_max = DataFrame(powermax)
    # df_min = DataFrame(powermin)

    # rename!(df_max, :dual => :powermax_dual)
    # rename!(df_min, :dual => :powermin_dual)

    # # Sort & merge
    # sort!(df_max, [:g, :t])
    # sort!(df_min, [:g, :t])

    # df = innerjoin(df_max, df_min, on = [:g, :t])

    # # ---------------------------------------------------------
    # # Save to CSV
    # # ---------------------------------------------------------
    # output_csv = joinpath(case, "Power_Duals.csv")
    # CSV.write(output_csv, df)

    # println("Wrote merged power duals to: ", output_csv)

    # === Build dual table ===
    # gen_list = myinputs["RESOURCES"]
    # num_gens = length(gen_list)

    # df = DataFrame(
    #     Index = Int[],
    #     Name = String[],
    #     cMaxCap = Float64[],
    #     cMinCapEnergyDuration = Float64[]
    # )

    # maxcap_con = haskey(EP, :cMaxCap) ? EP[:cMaxCap] : nothing
    # mindur_con = haskey(EP, :cMinCapEnergyDuration) ? EP[:cMinCapEnergyDuration] : nothing

    # for g in 1:num_gens
    #     res = gen_list[g]

    #     push!(df, (
    #         g,
    #         string(res.id),   # works for all thermals/VRE/hydro/storage

    #         get_dual(maxcap_con, g),
    #         get_dual(mindur_con, g)
    #     ))
    # end


    # # Write CSV
    # output_file = joinpath(case, "dual_table.csv")
    # CSV.write(output_file, df)

    # println("Dual table written to $output_file")


    ### Write outputs and optionally run MGA / Morris
    if has_values(EP)
        println("Writing Output")
        outputs_path = get_default_output_folder(case)
        elapsed_time = @elapsed write_outputs(EP, outputs_path, mysetup, myinputs)
        println("Time elapsed for writing: $elapsed_time seconds")

        if mysetup["ModelingToGenerateAlternatives"] == 1
            println("Starting MGA Iterations")
            mga(EP, case, mysetup, myinputs)
        end

        if mysetup["MethodofMorris"] == 1
            println("Starting Method of Morris sensitivity analysis")
            morris(EP, case, mysetup, myinputs, outputs_path, OPTIMIZER)
        end
    end
end

end # module