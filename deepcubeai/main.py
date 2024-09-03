import os
import subprocess
import sys


def main():
    dcai_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(dcai_dir)

    # Forward all arguments to the pipeline script, running from the new directory
    pipeline_script = os.path.join(dcai_dir, 'scripts', 'pipeline.sh')
    subprocess.run([pipeline_script] + sys.argv[1:], cwd=parent_dir)


if __name__ == "__main__":
    main()
