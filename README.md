Chinese-to-English Machine Translation Benchmark
================================================

Codes and pre-trained models for the Chinese-to-English machine translation benchmark (under construction).


Download Instruction
--------------------

In order to download, first clone this repository and the related submodules:

```
git clone https://github.com/nusnlp/c2e-mt-benchmark.git
cd c2e-mt-benchmark
git submodule update --init --recursive
```

Then, download and unpack the pre-trained models to the `models/` subdirectory:

```
cd models/
wget http://sterling8.d2.comp.nus.edu.sg/~christian/c2e-mt-benchmark/pretrained.tgz
tar -xvzf pretrained.tgz
cd ..
```


Publication
-----------

If you use the pre-trained models and settings from this repository, please cite the following paper:

Hadiwinoto, Christian and Ng, Hwee Tou (2018). Upping the ante: Towards a better benchmark for Chinese-to-English machine translation. To appear in *Proceedings of the 11th edition of the Language Resources and Evaluation Conference*. Miyazaki, Japan.
