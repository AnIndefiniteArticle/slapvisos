#! /bin/sh

pdflatex imagingoccs.tex
bibtex   imagingoccs.aux
pdflatex imagingoccs.tex
pdflatex imagingoccs.tex
