# Entrypoint script for docker container
# This script will be locatet /app/run.py

import subprocess
import os
import sys
def run(*popenargs, **kwargs):
    input = kwargs.pop("input", None)
    check = kwargs.pop("handle", False)

    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    process = subprocess.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr

# If no args are provided, explain
if len(sys.argv) < 3:
    print("Usage: python3 run.py <link> <target_folder>")
    sys.exit(1)

# Get link as sys.argv[1]
link = sys.argv[1]

# Get target folder as sys.argv[2]
target_folder = sys.argv[2]

# Get folder of this script
script_folder = os.path.dirname(os.path.realpath(__file__))

# Join script folder with download-piper-voices.py
download_script = os.path.join(script_folder, "download/download-piper-voices.py")

# Download the model
run(['python3', download_script, link, target_folder])

# Join target_folder with the model name
model_path = os.path.join(target_folder, "model.onnx")

# If the --speaker arg is provided, run the http server with the model and the speaker
if len(sys.argv) > 3:
    speaker = sys.argv[3]
    run(['python3', '-m', 'piper.http_server', '-m', model_path, '-s', speaker,'--cuda'])
    # sys.exit(0)
else:
    run(['python3', '-m', 'piper.http_server', '-m', model_path,'--cuda'])
    # sys.exit(0)