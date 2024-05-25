import csv

def generate_csv(outputs,filename = 'outputs.csv'):
    # Extract outputs
    #outputs = to.model.list_outputs(val=True, print_arrays=True)


    # Check if outputs are non-empty
    if not outputs:
        print("No outputs found!")
    else:
        # Print outputs for debugging
        for name, meta in outputs:
            value = meta['val']
            if value.size == 1:
                print(f"{name}: {value.item()}")
            else:
                print(f"{name}: {value.tolist()}")

        # Write outputs to CSV
        with open('outputs.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Field', 'Value'])

            for name, meta in outputs:
                value = meta['val']
                if value.size == 1:
                    writer.writerow([name, value.item()])
                else:
                    writer.writerow([name, value.tolist()])

    print("CSV file written successfully.")