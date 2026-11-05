"""Allow running SeekMorph via `python -m seekmorph`."""

import sys

from seekmorph.cli import main

if __name__ == "__main__":
    sys.exit(main())
