using YAML

file_path = "/home/kalvin/Documents/WEC-DECIDER/modules/CEM/data_east/cases/Case_year_2030_electrification_ref_carbon_constraint_med_wave_variability_Avail_zeta_0.05_omega_n_0.5_D_f_20_limited_100_wave_cost_Wave_400/settings/genx_settings.yml"
f = open(file_path)
data = YAML.load(f)
println(typeof(data))
println(data)
close(f)
