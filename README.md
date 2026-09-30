# LLM-based Low-Resource Machine Translation for Sorbian Languages with Restricted Resources

Master's thesis in Informatics, Technical University of Munich, 2026. Author: Ahmed Eltawel.

## Summary

This project studies how decoder-only large language models (LLMs) of up to 4B parameters
can be adapted to translate German into Upper Sorbian (hsb) and Lower Sorbian (dsb), two
West Slavic minority languages of eastern Germany with little parallel text. It follows the
setting of the WMT25 shared task on LLMs with Limited Resources for Slavic Languages.

The project:

- re-implements the TartuNLP and NRC recipes of the shared task in one training and
  evaluation pipeline;
- transfers the TartuNLP recipe to Qwen3.5 models at 0.8B, 2B and 4B parameters;
- tests additions to and removals from the training data at 0.8B: forward translation at
  three ratios, a different teacher model, new monolingual text, back-translation, Czech
  and Polish parallel data, removing similar Sorbian text with LaBSE, and removing part of
  the instruction data;
- scores every system with chrF++ (sacreBLEU) on the WMT25 development sets and tests
  differences with paired bootstrap resampling.

Main findings on the development sets: the choice and size of the pretrained model change
translation quality far more than any change to the training data. The 1.5B NRC
re-implementation and the 4B Qwen3.5 models score above the released 3B TartuNLP model in
both directions, and up to 9.59 % of the training tokens can be removed without a loss in
quality.

## Status

The repository is being prepared. The data, training and evaluation code will be added
after clean-up.
