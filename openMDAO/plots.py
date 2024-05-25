import matplotlib.pyplot as plt
def plot_openmdao_outputs(output_dict, design_var_names, objective_var_name):
    """
    Plots design variables and an objective variable from OpenMDAO outputs.

    Parameters:
    output_dict (dict): Dictionary of OpenMDAO outputs with variable names as keys and values as values.
    design_var_names (list): List of names of design variables to plot.
    objective_var_name (str): Name of the objective variable to plot.
    """
    # Extracting design variable values
    design_var_values = [output_dict[name] for name in design_var_names]

    # Extracting objective variable value
    obj_var_value = output_dict[objective_var_name]

    M_value = int(output_dict.get('ivc.M', 'N/A'))
    # Creating figure and axis
    fig, ax1 = plt.subplots()

    # Plotting design variables
    ax1.plot(design_var_names, design_var_values, 'b-', marker='o')
    ax1.set_xlabel('Design Variables')
    ax1.set_ylabel('Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating another y-axis for the objective variable
    ax2 = ax1.twinx()
    ax2.plot([objective_var_name], [obj_var_value], 'ro')
    ax2.set_ylabel(objective_var_name, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'Design Variables and Objective Variable m = {M_value}')
    plt.show()