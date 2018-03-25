#!/bin/bash
#Clean Wikipedia Data

 
wikipedia_dump_path=$1
../src/preprocess/wikiextractor/build/scripts-3.5/WikiExtractor.py -o data/enwiki ${wikipedia_dump_path}
