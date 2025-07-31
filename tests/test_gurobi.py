#!/usr/bin/env python3
"""
Test script to verify Gurobi installation in the devcontainer
"""

def test_gurobi_installation():
    """Test if Gurobi is properly installed and accessible"""
    
    import subprocess
    import os
    
    try:
        # Check if Gurobi environment variables are set
        gurobi_home = os.environ.get('GUROBI_HOME')
        if not gurobi_home:
            print("✗ GUROBI_HOME environment variable not set")
            return False
        
        print(f"✓ GUROBI_HOME set to: {gurobi_home}")
        
        # Check if Gurobi directory exists
        if not os.path.exists(gurobi_home):
            print(f"✗ Gurobi directory does not exist: {gurobi_home}")
            return False
        
        print("✓ Gurobi directory exists")
        
        # Check if gurobi_cl (command line tool) is available
        try:
            result = subprocess.run(['gurobi_cl', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ Gurobi command line tool accessible")
                print(f"  Version info: {result.stdout.strip()}")
                return True
            else:
                print(f"✗ Gurobi command line tool failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("✗ Gurobi command line tool timed out")
            return False
        except FileNotFoundError:
            print("✗ Gurobi command line tool not found in PATH")
            return False
            
    except Exception as e:
        print(f"✗ Gurobi test failed: {e}")
        return False

def test_julia_gurobi():
    """Test Julia Gurobi integration (if Julia is available)"""
    
    import subprocess
    
    try:
        # Check if Julia is available
        result = subprocess.run(['julia', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("✗ Julia not available, skipping Julia-Gurobi test")
            return False
        
        print("✓ Julia available")
        
        # Test Julia Gurobi package
        julia_test = '''
        try
            using Gurobi
            println("✓ Julia Gurobi package loaded successfully")
            
            # Try to create a Gurobi environment (this tests licensing)
            env = Gurobi.Env()
            println("✓ Gurobi environment created successfully")
            Gurobi.free_env(env)
            println("✓ Julia-Gurobi integration working")
            exit(0)
        catch e
            println("✗ Julia-Gurobi test failed: ", e)
            exit(1)
        end
        '''
        
        result = subprocess.run(['julia', '-e', julia_test], 
                              capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Julia-Gurobi test timed out")
        return False
    except Exception as e:
        print(f"✗ Julia-Gurobi test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Gurobi Installation Test")
    print("=" * 50)
    
    success1 = test_gurobi_installation()
    print()
    success2 = test_julia_gurobi()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Gurobi is ready for Julia.")
    elif success1:
        print("⚠️  Gurobi installed but Julia integration needs checking.")
        print("See .devcontainer/GUROBI_SETUP.md for help.")
    else:
        print("❌ Gurobi installation test failed.")
        print("See .devcontainer/GUROBI_SETUP.md for help.")
    print("=" * 50)
