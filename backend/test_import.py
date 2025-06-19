import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the required module
try:
    from langchain.agents.format_scratchpad import format_log_to_str
    print("Successfully imported format_log_to_str")
    print("Function signature:", format_log_to_str.__code__.co_varnames[:format_log_to_str.__code__.co_argcount])
except ImportError as e:
    print(f"Error importing format_log_to_str: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
    
    print("\nTrying to list langchain.agents package contents:")
    try:
        import pkgutil
        import langchain.agents
        print("langchain.agents package path:", langchain.agents.__file__)
        print("Submodules:", [name for _, name, _ in pkgutil.iter_modules(langchain.agents.__path__)])
    except Exception as e2:
        print(f"Error listing package contents: {e2}")
