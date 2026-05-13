module RunGenXCaseSimple

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

   # Comment these out to keep multistage active
   mysetup["MultiStage"] = 0
   mysetup["MultiStageSettingsDict"] = Dict()

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

    println("Writing the marginal thermal duals")
    #Prints the thermal generators max and min power constraints
    
    # open(joinpath(case, "thermal_duals_marginal.txt"), "w") do io
    #     println(io, "gen\ttime\tcMinPowerThermal\tcMaxPowerThermal\tcRampUpThermal\tcRampDownThermal")

    #     for t in 1:480
    #         for g in 4:27
    #             # Access named constraint references directly
    #             min_con = EP[:cMinPowerThermal][g,t]
    #             max_con = EP[:cMaxPowerThermal][g,t]
    #             ramp_up_con = EP[:cRampUpThermal][g,t]
    #             ramp_down_con = EP[:cRampDownThermal][g,t]

    #             # Get dual values
    #             min_val = try dual(min_con) catch; missing end
    #             max_val = try dual(max_con) catch; missing end
    #             ramp_up_val = try dual(ramp_up_con) catch; missing end
    #             ramp_down_val = try dual(ramp_down_con) catch; missing end

    #             if (min_val == 0.0 && max_val == 0.0 && ramp_up_val == 0.0 && ramp_down_val == 0.0)
    #                 println(io, g, "\t", t, "\t", min_val, "\t", max_val, "\t", ramp_up_val, "\t", ramp_down_val)
    #             end
    #         end
    #     end
    # end

    rows = DataFrame(
    gen = Int[],
    time = Int[],
    cMinPowerThermal = Any[],
    cMaxPowerThermal = Any[],
    cRampUpThermal = Any[],
    cRampDownThermal = Any[]
    )

    for t in 1:480
        for g in 4:27
            # Access named constraint references directly
            min_con = EP[:cMinPowerThermal][g,t]
            max_con = EP[:cMaxPowerThermal][g,t]
            ramp_up_con = EP[:cRampUpThermal][g,t]
            ramp_down_con = EP[:cRampDownThermal][g,t]

            # Get dual values
            min_val = try dual(min_con) catch; missing end
            max_val = try dual(max_con) catch; missing end
            ramp_up_val = try dual(ramp_up_con) catch; missing end
            ramp_down_val = try dual(ramp_down_con) catch; missing end

            if (min_val == 0.0 && max_val == 0.0 && ramp_up_val == 0.0 && ramp_down_val == 0.0)
                push!(rows, (
                    g,
                    t,
                    min_val,
                    max_val,
                    ramp_up_val,
                    ramp_down_val
                ))
            end
        end
    end

    CSV.write(joinpath(case, "thermal_duals_marginal.csv"), rows)

    tdm = CSV.read("data_east/cases/Case_year_2030_electrification_ref_carbon_constraint_med_wave_variability_Avail_zeta_0.05_omega_n_0.5_D_f_20_limited_100_wave_cost_Wave_400/thermal_duals_marginal.csv", DataFrame)

    gens_by_time = Dict{Int, Vector{Int}}()

    for t in sort(unique(tdm.time))
        gens_by_time[t] = tdm[tdm.time .== t, :gen]
    end

    filtered_gens_by_time = Dict{Int, Int}()

    #Here is the filter/priority process to choose coal>natural_gas>nuclear/other_peaker
    for t in sort(collect(keys(gens_by_time)))
        gens = gens_by_time[t]
        
        if 4 in gens
            filtered = 4

        elseif any(x -> x in gens, [5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27])
            filtered = minimum(intersect(gens, [5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27]))

        elseif any(x -> x in gens, [8, 9, 13, 14, 18])
            filtered = minimum(intersect(gens, [8, 9, 13, 14, 18]))

        else
            filtered = minimum(gens)

        end
    
        filtered_gens_by_time[t] = filtered
    end

    tdm_out = DataFrame(
    time = collect(keys(filtered_gens_by_time)),
    gen  = collect(values(filtered_gens_by_time))
    )

    sort!(tdm_out, :time)

    CSV.write("filtered_generators.csv", tdm_out)

    # rows = DataFrame(time = Int[], gen = Int[])
    # for t in 1:480
    #     gens = tdm[tdm.time .== t, :gen]

    #     for g in gens
    #         push!(rows, (t, g))
    #     end
    # end

    # CSV.write("TEST.csv", rows)

    
    # # -----------------------------
    # # Priority groups (explicit)
    # # -----------------------------
    # group1 = Set([4])

    # group2 = Set([
    #     5,6,7,10,11,12,15,16,17,19,20,21,22,23,24,25,26,27
    # ])

    # group3 = Set([8,9,13,14,18])

    # # -----------------------------
    # # Step 1: build dataset
    # # -----------------------------
    # rows = DataFrame(
    #     gen = Int[],
    #     time = Int[],
    #     cMinPowerThermal = Any[],
    #     cMaxPowerThermal = Any[],
    #     cRampUpThermal = Any[],
    #     cRampDownThermal = Any[]
    # )

    # for t in 1:480
    #     for g in 4:27
    #         push!(rows, (
    #             g,
    #             t,
    #             try dual(EP[:cMinPowerThermal][g,t]) catch; missing end,
    #             try dual(EP[:cMaxPowerThermal][g,t]) catch; missing end,
    #             try dual(EP[:cRampUpThermal][g,t]) catch; missing end,
    #             try dual(EP[:cRampDownThermal][g,t]) catch; missing end
    #         ))
    #     end
    # end

    # # -----------------------------
    # # Step 2: pick ONE generator per time step
    # # -----------------------------
    # filtered = DataFrame()

    # for t in unique(rows.time)
    #     sub = rows[rows.time .== t, :]

    #     chosen = nothing

    #     # 1. priority group 1 (gen 4)
    #     g1 = sub[in.(sub.gen, Ref(group1)), :]
    #     if nrow(g1) > 0
    #         chosen = g1[1, :]
    #     else
    #         # 2. group 2
    #         g2 = sub[in.(sub.gen, Ref(group2)), :]
    #         if nrow(g2) > 0
    #             chosen = g2[1, :]
    #         else
    #             # 3. group 3
    #             g3 = sub[in.(sub.gen, Ref(group3)), :]
    #             if nrow(g3) > 0
    #                 chosen = g3[1, :]
    #             end
    #         end
    #     end

    #     push!(filtered, chosen)
    # end

    # sort!(filtered, :time)

    # # -----------------------------
    # # Step 3: write CSV
    # # -----------------------------
    # CSV.write(joinpath(case, "thermal_duals_marginal_filtered.csv"), filtered)

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