# Prefix Beam Search
Code for prefix beam search tutorial by @labodk

Link: https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

### Code
This repository contains two files with Python code:

* `prefix_beam_search.py` contains all the code that is explained in the tutorial. I.e., the actual prefix beam search algorithm. 
* `test.py` will load a language model, perform beam search on three examples and print the result along with the output from a greedy decoder for comparison.

### Examples
The `examples` folder contains three examples of CTC output (2D NumPy arrays) from a CNN-based acoustic model. The model is trained on the LibriSpeech corpus (http://www.openslr.org/12). When executing `test.py` you should get the following output:

```
examples/example_1518.p
BEFORE: 0.00016sec, mister qualter as the apostle of the middle classes and we re glad twelcomed his gospe
AFTER(prob): 0.18531sec, mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
AFTER(logp): 0.73244sec, mister quilter is the apostle of the middle classes and we are glad to welcome his gospel


examples/example_99.p
BEFORE: 0.0001sec, but no ghoes tor anything else appeared upon the angient wall
AFTER(prob): 0.15459sec, but no ghost or anything else appeared upon the ancient walls
AFTER(logp): 0.58308sec, but no ghost or anything else appeared upon the ancient walls


examples/example_2002.p
BEFORE: 9e-05sec, alloud laugh followed at chunkeys expencs
AFTER(prob): 0.16504sec, a loud laugh followed at chunkys expense
AFTER(logp): 0.5361sec, a loud laugh followed at chunkys expense
```

Notice that each of these examples are handpicked. Thus, the transcript resulting from the prefix beam search is also the true transcript.

### Language Model
The `language_model.p` contains a dictionary mapping between all relevant prefixes queried during decoding of the three examples and the corresponding language model probabilities. Thus, this "language model" will only work for the three provided examples. The original language model file was too large to upload here. However, a range of similar pre-trained models can be found on the LibriSpeech website (http://www.openslr.org/11). The original model used in this tutorial was trained with the KenLM Language Model Toolkit (https://kheafield.com/code/kenlm/) on the additional language modeling data of the LibriSpeech corpus.

### Dependencies

* `numpy`
