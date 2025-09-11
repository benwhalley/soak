rm -rf dist build *.egg-info
python -m build
twine check dist/*

rm -rf dist build *.egg-info
python -m build && twine check dist/* \
  && python -m pip install --force-reinstall dist/*.whl \
  && python -c "import chatter, importlib.resources as r; print('OK', yourpkg.__version__)"
  


twine upload dist/*