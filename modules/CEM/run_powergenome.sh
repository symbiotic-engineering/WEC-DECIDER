# run sweep
#run_powergenome_multiple -sf ./data_CA/test_settings.yml     -rf ./data_CA/results

#echo "PowerGenome ran for CA."
echo "======================================================================="

run_powergenome_multiple -sf ./data_east/settings/ -rf ./data_east/results
for i in ./data_east/results/Case_*; do
    cp -r ./template/settings/ "$i/Inputs/Inputs_p1/"
done


echo "PowerGenome ran for NE."