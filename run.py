# Entrypoint script for docker container
# This script will be locatet /app/run.py

import subprocess
import os
import sys
if sys.version_info.major >= 3:
    from subprocess import CompletedProcess
else:
    # Add a polyfill to Python 2
    class CompletedProcess:

        def __init__(self, args, returncode, stdout=None, stderr=None):
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

        def check_returncode(self):
            if self.returncode != 0:
                err = subprocess.CalledProcessError(self.returncode, self.args, output=self.stdout)
                raise err
            return self.returncode

    def sp_run(*popenargs, **kwargs):
        input = kwargs.pop("input", None)
        check = kwargs.pop("handle", False)
        if input is not None:
            if 'stdin' in kwargs:
                raise ValueError('stdin and input arguments may not both be used.')
            kwargs['stdin'] = subprocess.PIPE
        process = subprocess.Popen(*popenargs, **kwargs)
        try:
            outs, errs = process.communicate(input)
        except:
            process.kill()
            process.wait()
            raise
        returncode = process.poll()
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, popenargs, output=outs)
        return CompletedProcess(popenargs, returncode, stdout=outs, stderr=errs)

    subprocess.run = sp_run
    # ^ This polyfill allows it work on Python 2 or 3 the same way

# If no args are provided, explain
if len(sys.argv) < 3:
    print("Usage: python run.py <link> <target_folder>")
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
subprocess.run(['python', download_script, link, target_folder])

# Join target_folder with the model name
model_path = os.path.join(target_folder, "model.onnx")

# If the --speaker arg is provided, run the http server with the model and the speaker
if len(sys.argv) > 3:
    speaker = sys.argv[3]
    subprocess.run(['python', '-m', 'piper.http_server', '-m', model_path, '-s', speaker,'--cuda'])
    # sys.exit(0)
else:
    subprocess.run(['python', '-m', 'piper.http_server', '-m', model_path,'--cuda'])
    # sys.exit(0)