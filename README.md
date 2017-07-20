# Multilingual Modal Sense Classification using a Convolutional Neural Network. 
Does a modal verb express possibility, permission or capability?

Ana Marasovic and Anette Frank. (2016):
[Multilingual Modal Sense Classification using a Convolutional Neural Network](http://www.aclweb.org/anthology/W/W16/W16-1613.pdf). Proceedings of the 1st Workshop on Representation Learning for NLP, Berlin, Germany, pp. 111--120, Association for Computational Linguistics.
<p align="center"><img src="figs/mr_lstm.png" width="400" align="middle" alt="LSTM-Siamese mention-ranking model"></p>

## Requirements

- [tensorflow 0.12](https://www.tensorflow.org/versions/r0.12/)

## Notes

Internally we used following abbreviations,
- balance, unbalance - dataset was balanced or not
- classifier1, classifier2, classifier3 - MPQA, EPOS+MPQA, EPOS
Please refer to the paper what MPQA, EPOS+MPQA, EPOS are.

Not everything is provided with in the repo. More precisely,
1) there is no code for loading dependency-based embeddings (trivial)
2) words which don?t have a pre-trained embedding are not randomly initializing with a random vector sampled from uniform distribution which has the same variance as the pre-trained embeddings (trivial)
3) there is no code for micro-average calculation (trivial)
4) no code for baselines 
5) no code for significance testing (trivial)
6) no code for semantic feature detectors
7) no code for WSD

The code is re-factorize after the submission and results could differ from the reported results. The script how to get individual results and macro-average can be found in bash_scripts/epos_eng.sh. If you get results that differ in large margins, please contact me.

The code is hard-coded with paths to the English datasets, change paths for experiments on German data.


## Reference

If you make use of the contents of this repository, we appreciate citing [the following paper](http://aclweb.org/anthology/W/W16/W16-1613.pdf):

```
@inproceedings{marasovic2016msc,
  title={{Multilingual Modal Sense Classification using a Convolutional Neural Network}},
  author={Marasovic, Ana and Frank, Anette},
  booktitle={Proceedings of the 1st Workshop on Representation Learning for NLP},
  year={2016}
}
```

