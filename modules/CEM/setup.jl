using Pkg

project_path = @__DIR__
Pkg.activate(project_path)

Pkg.instantiate() # This is necessary every time Project.toml is modified

function ensure_gurobi_works()
    try
        @eval using Gurobi
        redirect_stdout(devnull) do
            model = Gurobi.Env()
            Gurobi.free_env(model)
        end
    catch
        # This is only necessary once per Julia/Gurobi installation
        Pkg.build("Gurobi")
    end
end

ensure_gurobi_works()
