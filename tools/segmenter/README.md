Instructions for using the Chinese word segementer
--------------------------------------------------

Jin Kiat Low
Hwee Tou Ng
National University of Singapore

21 August 2006


This Chinese word segmenter is provided for use in research or
non-commerical purpose. If your publication uses this segmenter,
please give a citation to the following paper which describes the
details of the segmenter:

Low, Jin Kiat, & Ng, Hwee Tou, & Guo, Wenyuan (2005). A Maximum
Entropy Approach to Chinese Word Segmentation. Proceedings of the
Fourth SIGHAN Workshop on Chinese Language Processing.
(pp. 161-164). Jeju Island, Korea.
http://www.comp.nus.edu.sg/~nght/pubs/sighan05.pdf


Requirement
-----------

* Python >= 2.6


Running the Chinese word segmenter
----------------------------------

python2 segment.py -m ctbModel -l lex.GB.txt < infile > outfile

The following model file is available:

ctbModel 

They were trained on CTB training data of SIGHAN bakeoff 1. The
encoding used is gb.



