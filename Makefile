SHELL := /bin/bash
export TEXINPUTS := .//:

all: build
.PHONY: clean 

clean:
	rm -rf *.blg 
	rm -rf *.out 
	rm -rf *.bbl 
	rm -rf *.log
	rm -rf *.ind
	rm -rf *.ilg
	rm -rf *.lot
	rm -rf *.lof
	rm -rf *.idx
	rm -rf *.aux
	rm -rf *.toc
	rm -f thesis.pdf
	rm -f output.pdf
	rm -f smalloutput.pdf

build:
	# export TEXINPUTS=".//:"
	echo $$TEXINPUTS
	pdflatex -shell-escape -jobname output -draftmode thesis.tex -interaction=batchmode
	pdflatex -shell-escape -jobname output thesis.tex -interaction=batchmode
	bibtex output
	makeindex thesis.tex
	for FILE in `ls feynman_diagrams/*.mp`; do mpost $$FILE; done
	mv *.{1,t1} feynman_diagrams/
	pdflatex -shell-escape -jobname output -draftmode thesis.tex -interaction=batchmode
	pdflatex -shell-escape -jobname output thesis.tex
	mkdir -p tmp
	mv *.{ind,blg,out,bbl,log,ilg,aux,toc} tmp/

# single spaced small version
small:
	pdflatex -shell-escape -jobname smalloutput "\def\myownflag{}\input{thesis}" -interaction=batchmode
	pdflatex -shell-escape -jobname smalloutput "\def\myownflag{}\input{thesis}" -interaction=batchmode
	makeindex thesis.tex
	bibtex smalloutput
	pdflatex -shell-escape -jobname smalloutput "\def\myownflag{}\input{thesis}" -interaction=batchmode
	pdflatex -shell-escape -jobname smalloutput "\def\myownflag{}\input{thesis}"
	mkdir -p tmp
	mv -f *.{ind,blg,out,bbl,log,ilg,aux,toc} tmp/
