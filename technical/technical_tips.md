# Technical tips

## Jupyter notebook & venv

See `jupyter_backround.sh` for my script that activates the environment and starts the jupyter kernel.  

Note that I have it start @ port 8889, since I want 8888 free for cluster connections, for example.
Then simply go to http://localhost:8889/ (bookmark this) - you should see `rule-env` in the top-right and the option to create a new environment.
You can also change kernel in the kernel menu.

<!-- [Start IPython with venv](https://stackoverflow.com/questions/20327621/calling-ipython-from-a-virtualenv) -->