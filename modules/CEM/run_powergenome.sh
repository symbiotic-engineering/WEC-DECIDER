script_dir=$(dirname "$0")

# run sweep
#run_powergenome_multiple -sf ./data_CA/test_settings.yml     -rf ./data_CA/cases

#echo "PowerGenome ran for CA."
echo "======================================================================="

run_powergenome_multiple -sf $script_dir/data_east/settings/ -rf $script_dir/data_east/cases
for i in $script_dir/data_east/cases/Case_*; do
    cp -r $script_dir/template/settings/ "$i"
done


echo "PowerGenome ran for NE."