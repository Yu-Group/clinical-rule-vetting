#! /bin/bash
cd /home/andrej/myStuff/berkeley/STAT215A/assignments/final
source rule-env/bin/activate
jupyter notebook --no-browser --notebook-dir=/home/andrej/myStuff/berkeley/STAT215A/assignments/final --log-level='CRITICAL' --port 8889 &