Chinese-to-English Machine Translation Benchmark
================================================

Codes and pre-trained models for the Chinese-to-English machine translation benchmark.


Setup Instructions
------------------

Fistly, clone this repository and the related submodules:

```
git clone https://github.com/nusnlp/c2e-mt-benchmark.git
cd c2e-mt-benchmark
git submodule update --init --recursive
```

Secondly, go to each subdirectories under `tools/*` and follow the setup/installation instructions accordingly.

Finally, download and unpack the pre-trained models to the `models/` subdirectory:

```
cd models/
wget http://sterling8.d2.comp.nus.edu.sg/~christian/c2e-mt-benchmark/pretrained.tgz
tar -xvzf pretrained.tgz
cd ..
```


Running Instructions
--------------------

(under construction)

The input is a plain text file containing Chinese sentences, one sentence per line. The input file is passed through the following pipeline:

1. Chinese word segmentation, by running `scripts/segment.sh < input > input.seg`
2. Translation (ensure that Theano flags are set accordingly)
   * without reranking: (to be written)
   * with re-ranking: `scripts/translate-rerank.sh nist input.seg output`
3. Recasing and post-processing, by running

Publication
-----------

If you use the pre-trained models and settings from this repository, please cite the following paper:

Hadiwinoto, Christian and Ng, Hwee Tou (2018). Upping the ante: Towards a better benchmark for Chinese-to-English machine translation. To appear in *Proceedings of the 11th edition of the Language Resources and Evaluation Conference*. Miyazaki, Japan.
