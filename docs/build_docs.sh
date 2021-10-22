PKG=rulevetting
cd ../$PKG
pdoc --html . --output-dir ../docs --template-dir .
cp -rf ../docs/$PKG/* ../docs/
rm -rf ../docs/$PKG
cd ../docs
python3 style_docs.py