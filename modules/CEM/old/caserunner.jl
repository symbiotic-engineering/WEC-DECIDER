using CSV
using DataFrames
using YAML

# Global constants accessible to all functions
caserunner_specialstring() = "SPECIAL"
caserunner_templatefolder() = "inputs"
caserunner_jobscript() = "jobscript.sh"
main_case_folder() = "Cases"
settingsfolder() = "Settings"
results_name() = "Results"

csv2dataframe(path::AbstractString) = CSV.read(path, header=1, DataFrame)
dataframe2csv(df::DataFrame, path::AbstractString) = CSV.write(path, df)
yml2dict(path::AbstractString) = YAML.load_file(path)
dict2yml(d::Dict, path::AbstractString) = YAML.write_file(path, d)

function run_caserunner(pg_id, location)
    caserunner_replacementscsv = "replacements_" * location * ".csv"
    replacements_df = csv2dataframe(caserunner_replacementscsv)

    case_folder_name(i::Integer) = "case_" * string(i) * "_" * pg_id
    case_folder_path(i::Integer) = joinpath(main_case_folder(), case_folder_name(i))

    launch_new_cases(case_folder_name, case_folder_path, replacements_df)
end

function run_job(i, joblocation, case_folder_path)
    if joblocation == "BATCH"
        run_job_batch(i, case_folder_path)
    elseif joblocation == "SEQUENTIAL"
        run_job_sequential(i, case_folder_path)
    else
        error("The variable joblocation should be `BATCH` or `SEQUENTIAL`")
    end
end

function run_job_sequential(i, case_folder_path)
    origdir = pwd()
    path = joinpath(case_folder_path(i), "Run.jl")
    include(path)
    cd(origdir)
end

function run_job_batch(i, case_folder_path)
    origdir = pwd()
    path = case_folder_path(i)
    cd(path)
    run(`sbatch jobscript.sh`)
    cd(origdir)
end

function files_to_check()
    csvs = csv_files_to_check()
    ymls = yaml_files_to_check()
    return vcat(csvs, ymls)
end

function redetermine_file_type(path)
    last4 = path[end-3:end]
    if last4 == ".csv"
        return "csv"
    elseif last4 == ".yml"
        return "yml"
    else
        error("Unknown file type")
    end
end

function csv_files_to_check()
    all_entries = readdir(caserunner_templatefolder());
    return [f for f in all_entries if f[end-3:end] == ".csv"]
end

function yaml_files_to_check()
    path = joinpath(caserunner_templatefolder(), settingsfolder())
    all_entries = readdir(path);
    return [joinpath(settingsfolder(), f) for f in all_entries if f[end-3:end] == ".yml"]
end

function isspecialkey(s::AbstractString)
    if length(s) < 13
        return false
    end
    test1 = s[begin:2] == "__"
    test2 = s[end-1:end] == "__"
    if !test1 || !test2
        return false
    end
    elements = split(s, "_")
    nonblank_elements = [i for i in elements if i != ""]
    test3 = nonblank_elements[1] == caserunner_specialstring()
    test4 = length(nonblank_elements) == 2
    return test3 && test4
end

function extractspecialkey(s::AbstractString)
    elements = split(s, "_")
    nonblank_elements = [i for i in elements if i != ""]
    return nonblank_elements[2]
end

function check_element(e)
    if e isa AbstractString && isspecialkey(e)
        return String[extractspecialkey(e)]
    else
        return String[]
    end
end

function check_datastructure(d::Dict)
    key_fields_found = String[]
    for (key, value) in d
        key_fields_found = vcat(key_fields_found, check_element(value))
    end
    return key_fields_found
end

function check_datastructure(df::DataFrame)
    key_fields_found = String[]
    for c in eachcol(df)
        for r in c
            key_fields_found = vcat(key_fields_found, check_element(r))
        end
    end
    return key_fields_found
end

function check_file(name)
    path = joinpath(caserunner_templatefolder(), name)
    if !isfile(path)
        error("$path is not a file and/or does not exist")
    else
        ftype = redetermine_file_type(name)
        if ftype == "csv"
            data = csv2dataframe(path)
        elseif ftype == "yml"
            data = yml2dict(path)
        else
            error("Checking unknown file type.")
        end
        return check_datastructure(data)
    end
end

function flag_nonmatchingkeys(key_fields_found::Vector{String}, replacements::Vector{String})
    kfs = Set(key_fields_found)
    replacementnames = Set(replacements)
    diff1 = setdiff(replacementnames, kfs)
    diff2 = setdiff(kfs, replacementnames)

    if length(diff1) > 0
        error("""
            Not all special key fields listed in replacements.csv
            were found in the template files. In particular,
            $diff1 were not found.
            """)
    elseif length(diff2) > 0
        error("""
            Not all special key fields found in the template
            files were listed as columns in replacements.csv.
            In particular, $diff2 were not found.
            """)
    end
end

function flag_badkeys(key_fields_found::Vector{String}, replacements::Vector{String})
    flag_nonmatchingkeys(key_fields_found, replacements)
end

function check_files()
    key_fields_found = String[]
    files_with_keys = String[]
    for f in files_to_check()
        results = check_file(f)
        if length(results) > 0
            push!(files_with_keys, f)
        end
        key_fields_found = vcat(key_fields_found, results)
    end

    flag_badkeys(key_fields_found, get_replacement_names())

    return files_with_keys, key_fields_found
end

function ensure_main_cases_folder()
    if !isdir(main_case_folder())
        mkdir(main_case_folder())
    end
end

function case_folder_exists(i::Integer, case_folder_name)
    path = joinpath(main_case_folder(), case_folder_name(i))
    return isdir(path)
end

function case_folder_complete(i::Integer, case_folder_name)
    path = joinpath(main_case_folder(), case_folder_name(i), results_name())
    return isdir(path)
end

function copy_to_new_case_folder(i::Integer, case_folder_name)
    ensure_main_cases_folder()
    path = joinpath(main_case_folder(), case_folder_name(i))
    cp(caserunner_templatefolder(), path)
end

function get_replacement_names(df::DataFrame)
    names(df[:, Not([:Case, :Notes])])
end

function get_specific_replacements(df::DataFrame, i::Integer)
    return df[df[:, :Case] .== i, Not([:Case, :Notes])][1,:]
end

function number_of_replacement_cases(df::DataFrame)
    return size(df)[1]
end

function replace_elements!(df::DataFrame, replacements::Dict)
    for ci in 1:size(df)[2], ri in 1:size(df)[1]
        element = df[ri, ci]
        if element isa AbstractString && isspecialkey(element)
            df[ri, ci] = string(replacements[element])
        end
    end
end

function replace_elements!(d::Dict, replacements::Dict)
    for (key, value) in d
        if value in keys(replacements)
            d[key] = replacements[value]
        end
    end
end

function dataframerow2dict(r::DataFrameRow)
    replnames = string_to_specialkey.(names(r))
    replvalues = values(r)
    return Dict(zip(replnames, replvalues))
end

string_to_specialkey(s::AbstractString) = "__" * caserunner_specialstring() * "_" * s * "__"

function replace_keys_in_csv_file(path::AbstractString, replacements::DataFrameRow)
    s = csv2dataframe(path)
    replace_elements!(s, dataframerow2dict(replacements))
    dataframe2csv(s, path)
end

function replace_keys_in_yml_file(path::AbstractString, replacements::DataFrameRow)
    s = yml2dict(path)
    replace_elements!(s, dataframerow2dict(replacements))
    dict2yml(s, path)
end

function replace_keys_in_file(path::AbstractString, replacements::DataFrameRow)
    ftype = redetermine_file_type(path)
    if ftype == "csv"
        replace_keys_in_csv_file(path, replacements)
    elseif ftype == "yml"
        replace_keys_in_yml_file(path, replacements)
    else
        error("Unknown file type")
    end
end

function replace_keys_in_folder(i::Int, replacements::DataFrameRow, files_with_keys::Vector{String}, case_folder_path)
    folder = case_folder_path(i)
    for f in files_with_keys
        path = joinpath(folder, f)
        replace_keys_in_file(path, replacements)
    end
end

function launch_new_case(i::Integer, df::DataFrame, files_with_keys::Vector{String}, case_folder_name, case_folder_path)
    copy_to_new_case_folder(i, case_folder_name)
    replacements = get_specific_replacements(df, i)
    replace_keys_in_folder(i, replacements, files_with_keys, case_folder_path)
    run_job(i, "SEQUENTIAL", case_folder_path)
end

function launch_new_cases(case_folder_name, case_folder_path, replacements_df)
    files_with_keys, keys_found = check_files()
    cases = replacements_df[:, :Case]
    for c in cases
        if case_folder_complete(c, case_folder_name)
            println("Case $c complete; skipping.")
        elseif case_folder_exists(c, case_folder_name)
            println("Case $c exists; skipping.")
        else
            println("Case $c now creating")
            launch_new_case(c, replacements_df, files_with_keys, case_folder_name, case_folder_path)
        end
    end
end
