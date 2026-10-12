"""Run the local dataset integrity and exact-duplicate audit."""
from run_local import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--stage", "verify"]
    main()
