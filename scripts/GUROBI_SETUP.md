# Gurobi Setup for WEC-DECIDER Devcontainer

This devcontainer includes Gurobi Optimizer integration for use with Julia. Here are the setup options:

## Option 1: Using Gurobi License

If you have a Gurobi license:

1. **Academic License**: Get a free academic license from https://www.gurobi.com/academia/academic-program-and-licenses/
2. **Commercial License**: Use your existing Gurobi license

### Setup Steps:

1. **License File Method**:
   ```bash
   # Create license directory in your home folder
   mkdir -p ~/.gurobi
   
   # Copy your license file
   cp /path/to/your/gurobi.lic ~/.gurobi/
   ```

2. **Environment Variable Method**:
   ```bash
   # Set your license ID as environment variable
   export GRB_LICENSEID="your-license-id"
   ```

3. **License Server Method** (for network licenses):
   ```bash
   # Set license server
   export GRB_LICENSE_SERVER="server-address:port"
   ```

## Option 2: Gurobi Web License Service (WLS)

For the free limited version:

1. Create account at https://www.gurobi.com/
2. Get a free WLS license
3. Set environment variable:
   ```bash
   export GRB_WLSACCESSID="your-wls-access-id"
   export GRB_WLSSECRET="your-wls-secret"
   ```

## Testing Installation

Once the devcontainer is running, test Gurobi:

```bash
# Test command line access
gurobi_cl --version

# Run the test script
python tests/test_gurobi.py

# Or run the license setup script manually if needed
scripts/setup_gurobi_license.sh
```

## Integration with Existing Code

Your project uses Gurobi in the Julia CEM module:

```julia
using Gurobi

# Create a Gurobi environment
env = Gurobi.Env()

# Use with GenX
run_genx_case!(case_dir, Gurobi.Optimizer)
```

The Gurobi installation provides:
- Command line interface (`gurobi_cl`)
- C/C++ libraries for Julia integration
- Environment variables for license management

## Troubleshooting

- **License not found**: Check that `GRB_LICENSE_FILE` points to your license file
- **Network issues**: Ensure license server is accessible if using network licensing
- **Academic license**: Make sure you're on a university network or using VPN if required
- **Julia integration**: Run `julia -e "using Gurobi; println(\"OK\")"` to test Julia package

For more details, see: https://www.gurobi.com/documentation/
