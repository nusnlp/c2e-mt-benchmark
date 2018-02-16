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



Running the Chinese word segmenter
----------------------------------

A csh script cmdSeg is provided to run the segmenter:

cmdSeg modelFile testinFile encoding segmentedFile dictFile

where dictFile is an optional argument.

eg, cmdSeg ctbModel infile gb outfile lex.GB.txt

If the arugment dictFile is not provided, then only basic features are
used for word segmentation (i.e., without making use of any external
dictionary).

The following model file is available:

ctbModel 

They were trained on CTB training data of SIGHAN bakeoff 1, and MSR
training data of SIGHAN bakeoff 2, respectively.  They were of version
"V4", which means they combined the use of basic features, external
dictionary (the downloaded PKU dictionary), and additional training
corpora (of other segmentation standards) via example selection.  The
encoding used is gb.



