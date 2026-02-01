"""
Simple test script to verify app.py syntax and imports
Run: python test_app.py
"""

import sys
import ast

def check_syntax(filepath):
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✅ Syntax check passed: {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}:")
        print(f"   Line {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        return False

def check_imports():
    """Check if all required packages can be imported"""
    required_packages = [
        ('streamlit', 'st'),
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('plotly.express', 'px'),
        ('plotly.graph_objects', 'go'),
        ('sklearn.linear_model', 'LinearRegression'),
        ('sklearn.cluster', 'KMeans'),
        ('sklearn.preprocessing', 'StandardScaler'),
        ('sklearn.decomposition', 'PCA'),
        ('sklearn.metrics', 'mean_squared_error'),
        ('prophet', 'Prophet'),
        ('scipy.stats', 'stats'),
        ('configparser', 'ConfigParser'),
        ('subprocess', 'run'),
    ]
    
    print("\n📦 Checking imports...")
    all_ok = True
    
    for package, name in required_packages:
        try:
            if '.' in package:
                exec(f"from {package} import {name}")
            else:
                exec(f"import {package}")
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("🔍 Testing app.py...")
    print("-" * 50)
    
    # Check syntax
    syntax_ok = check_syntax('app.py')
    
    # Check imports
    import_ok = check_imports()
    
    print("-" * 50)
    if syntax_ok and import_ok:
        print("✅ All checks passed! Ready to run:\n   streamlit run app.py")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please fix errors above.")
        sys.exit(1)
