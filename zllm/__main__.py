"""Allow `python -m zllm`."""
from zllm.cli import main
import sys
sys.exit(main() or 0)
