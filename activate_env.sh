if [ ! -e env/bin/activate ]; then
  python3.9 -m venv env/ || return # exit on error
  source env/bin/activate
  echo "Activated virtual env"
  python -m pip install --upgrade pip wheel setuptools
  
else
  source env/bin/activate
  echo "Activated virtual env"
fi

python -m pip install -e .
