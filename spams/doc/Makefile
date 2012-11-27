OPT=-O article.hva macrocss.hva
#fancysection.hva
all:
	pdflatex doc_spams.tex
	bibtex doc_spams
	pdflatex doc_spams.tex
	pdflatex doc_spams.tex
hevea:
	hevea ${OPT} doc_spams.tex
	bibhva doc_spams
	hevea ${OPT} doc_spams.tex
	hevea ${OPT} doc_spams.tex
	hacha -tocbis doc_spams.html
	cp style.css doc_spams.css
	mv *.html html/
	mv *.gif html/
	mv *.css html/
	mv html/style.css ./
clean:
	rm -f *.aux *.bbl *.haux *.hbbl *.blg *.toc *.htoc *.log

