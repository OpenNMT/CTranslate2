#! /bin/bash

set -e
set -x

# force use pip < 24.1
python -m pip install 'pip<24.1'

# Install test rquirements
pip cache purge
pip --no-cache-dir install -r python/tests/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip uninstall -y ctranslate2

# Download test data
curl -o transliteration-aren-all.tar.gz https://opennmt-models.s3.amazonaws.com/transliteration-aren-all.tar.gz
tar xf transliteration-aren-all.tar.gz -C tests/data/models/
rm transliteration-aren-all.tar.gz

curl -O https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip
unzip opus-2020-02-26.zip -d tests/data/models/opus-mt-ende
rm opus-2020-02-26.zip
