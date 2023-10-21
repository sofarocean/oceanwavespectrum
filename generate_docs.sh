#!/bin/bash
doc_directory=docs
mkdir -p $doc_directory
pdoc src/roguewavespectrum -o $doc_directory --math
