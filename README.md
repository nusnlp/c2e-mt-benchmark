Chinese-to-English Machine Translation Benchmark
================================================

Codes and pre-trained models for the Chinese-to-English machine translation benchmark.


Setup
-----

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
wget http://sterling8.d2.comp.nus.edu.sg/~christian/c2e-mt-benchmark/pretrained.tar.gz
tar -xvzf pretrained.tar.gz
cd ..
```

Translation
-----------

The input is a plain text file containing Chinese sentences, one sentence per line. The input file is passed through the following pipeline:

1. Chinese word segmentation, by running `scripts/segment.sh < input > input.seg`
2. Translation (ensure that Theano flags are set as environment variables, replace `nist` with `unpc` for models trained on UN Parallel Corpus)
   * without re-ranking: `scripts/translate-norerank.sh nist input.seg output [device(s)]`, where the device(s) include "gpu0", "gpu0 gpu1", or the default "cpu"
   * with re-ranking: `scripts/translate-rerank.sh nist input.seg output [device(s)]`
3. Recasing, by running `scripts/recase.sh < output > output.rc`
4. Detokenization, by running `perl scripts/detokenizer.perl -l en < output.rc > output.detok`

Publication
-----------

If you use the pre-trained models and settings from this repository, please cite the following paper:

Hadiwinoto, Christian and Ng, Hwee Tou (2018). Upping the ante: Towards a better benchmark for Chinese-to-English machine translation. To appear in *Proceedings of the 11th edition of the Language Resources and Evaluation Conference*. Miyazaki, Japan.
