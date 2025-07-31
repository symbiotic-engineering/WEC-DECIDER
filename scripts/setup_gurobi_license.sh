#!/bin/bash
# Gurobi license setup script for both CI and local development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LICENSE_FILE=""

# Function to check if Gurobi license is working
check_gurobi_license() {
    echo "Checking Gurobi license..."
    
    # Try using the command line tool first
    if gurobi_cl --version > /dev/null 2>&1; then
        # Create a simple test file
        cat > /tmp/gurobi_test.lp << EOF
Minimize
 obj: x + y
Subject To
 c1: 2 x + y <= 3
 c2: x + 2 y <= 3
Bounds
 x >= 0
 y >= 0
End
EOF
        
        if gurobi_cl /tmp/gurobi_test.lp > /dev/null 2>&1; then
            echo "✅ Gurobi license is working"
            rm -f /tmp/gurobi_test.lp
            return 0
        else
            echo "❌ Gurobi license check failed (solver error)"
            rm -f /tmp/gurobi_test.lp
            return 1
        fi
    else
        echo "❌ Gurobi command line tool not accessible"
        return 1
    fi
}

# Function to set up WLS license from environment variables
setup_wls_from_env() {
    if [[ -n "${WLSACCESSID:-}" && -n "${WLSSECRET:-}" && -n "${LICENSEID:-}" ]]; then
        echo "Setting up WLS license from environment variables..."
        LICENSE_FILE="/tmp/gurobi_env.lic"
        cat > "$LICENSE_FILE" << EOF
# Gurobi WLS license file (generated from environment)
WLSACCESSID=${WLSACCESSID}
WLSSECRET=${WLSSECRET}
LICENSEID=${LICENSEID}
EOF
        export GRB_LICENSE_FILE="$LICENSE_FILE"
        echo "License file created at: $LICENSE_FILE"
        return 0
    fi
    return 1
}

# Function to find existing license files
find_existing_license() {
    local possible_locations=(
        "${HOME}/.gurobi/gurobi.lic"
        "${SCRIPT_DIR}/../.github/workflows/gurobi.lic"
        "${GRB_LICENSE_FILE:-}"
        "/opt/gurobi/gurobi.lic"
    )
    
    for location in "${possible_locations[@]}"; do
        if [[ -n "$location" && -f "$location" ]]; then
            echo "Found existing license file: $location"
            export GRB_LICENSE_FILE="$location"
            return 0
        fi
    done
    return 1
}

# Main setup logic
main() {
    echo "=== Gurobi License Setup ==="
    
    # Try different license setup methods in order of preference
    
    # 1. Check if license is already working
    if check_gurobi_license; then
        echo "✅ Gurobi license already configured and working"
        return 0
    fi
    
    # 2. Try to set up from environment variables (CI case)
    if setup_wls_from_env && check_gurobi_license; then
        echo "✅ WLS license configured from environment variables"
        return 0
    fi
    
    # 3. Try to find existing license files
    if find_existing_license && check_gurobi_license; then
        echo "✅ Using existing license file: $GRB_LICENSE_FILE"
        return 0
    fi
    
    # 4. Provide setup instructions
    echo ""
    echo "❌ No working Gurobi license found. Please set up a license:"
    echo ""
    echo "For CI/GitHub Actions:"
    echo "  Set repository secrets: WLSACCESSID, WLSSECRET, LICENSEID"
    echo ""
    echo "For local development:"
    echo "  1. Academic license: https://www.gurobi.com/academia/"
    echo "  2. Copy license to: ~/.gurobi/gurobi.lic"
    echo "  3. Or set environment: export GRB_LICENSE_FILE=/path/to/gurobi.lic"
    echo ""
    echo "For WLS (Web License Service):"
    echo "  export WLSACCESSID=your_access_id"
    echo "  export WLSSECRET=your_secret"
    echo "  export LICENSEID=your_license_id"
    echo ""
    
    return 1
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
