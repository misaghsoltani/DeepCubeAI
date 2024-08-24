DCAI_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-$0}) && pwd)

# Check if $DCAI_DIR is already at the beginning of PYTHONPATH
if [[ "$PYTHONPATH" != "$DCAI_DIR:"* ]]; then
    # Remove $DCAI_DIR from PYTHONPATH if it exists anywhere
    PYTHONPATH=$(echo "$PYTHONPATH" | sed -e "s|^$DCAI_DIR:*||" -e "s|:$DCAI_DIR||")

    # Add $DCAI_DIR to the beginning of PYTHONPATH
    export PYTHONPATH="$DCAI_DIR:$PYTHONPATH"
fi
