# GenX Fork for Wave Energy
All the files for wave energy are in `Example_Systems/RealSystemExample/ISONE_Singlezone/`.

0. Clone this repo, setup conda environment with `environment.yml`, install Julia and Gurobi 
1. Configure `h5pyd` according to the instructions [here](https://mhkit-software.github.io/MHKiT/WPTO_hindcast_example.html)
1. First run `make_batch_csv.ipynb` which should generate `replacements.csv`
2. Then run `caserunner.jl` (see the bottom of `make_batch_csv.ipynb` for julia commands) to run the optimization
import package genX
Package activate Gen x
package instantiate
Package st to check proper packages
cd to proper directory
obtain gurobi licence and then pkg.build("Gurobi")
include caserunner
3. Finally run `analyze_results.ipynb`

See the report `report.pdf` for details on the methodology and results.

Author: @rebeccamccabe