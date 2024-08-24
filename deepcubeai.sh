DCAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $DCAI_DIR

# Setup PYTHONPATH
DCAI_SETUP_DIR="$DCAI_DIR/setup.sh"
source "$DCAI_SETUP_DIR"

# Forward all arguments to the pipeline script
bash "$DCAI_DIR/deepcubeai/scripts/pipeline.sh" "$@"
