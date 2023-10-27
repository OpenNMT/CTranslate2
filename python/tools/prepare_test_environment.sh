#! /bin/bash

set -e
set -x

# Install test rquirements
pip --no-cache-dir install -r python/tests/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip uninstall -y ctranslate2

# Download test data
curl -o transliteration-aren-all.tar.gz https://opennmt-models.s3.amazonaws.com/transliteration-aren-all.tar.gz
tar xf transliteration-aren-all.tar.gz -C tests/data/models/
rm transliteration-aren-all.tar.gz

curl -O https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip
unzip opus-2020-02-26.zip -d tests/data/models/opus-mt-ende
rm opus-2020-02-26.zip

curl https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav --output tests/data/audio/test.wav
