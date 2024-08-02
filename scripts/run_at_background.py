import os
import sys
os.system("screen -S ar '" +
          sys.executable + " automatic_artifact_rejection.py && " +
          "' &")