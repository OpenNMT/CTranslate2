#!/bin/bash
python3 -c 'import sys; import pyonmttok; pyonmttok.Tokenizer("none", sp_model_path="/workspace/sentencepiece.model").detokenize_file(sys.argv[1], sys.argv[2])' $1 $2
