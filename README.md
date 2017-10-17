# prefix-beam-search
Code for prefix beam search tutorial by @labodk
Link: TODO: ADD LINK

### Code
This repository contains two files with Python code:

* `prefix_beam_search.py` contains all the code that is explained in the tutorial 
* `run_examples.py` will load a laguage model, perform beam search on three examples and print the result along with the output from a greedy decoder for comparison.

### Examples
The `examples` folder contains three examples of CTC output (2D NumPy arrays) from a CNN-based acoustic model. The model is trained on the LibriSpeech corpus. When running `run_examples.py` you should get the following output:

```
examples/example_2002.p

BEFORE:
alloud laugh followed at chunkeys expencs

AFTER:
a loud laugh followed at chunkys expense

examples/example_99.p

BEFORE:
but no ghoes tor anything else appeared upon the angient wall

AFTER:
but no ghost or anything else appeared upon the ancient walls

examples/example_1518.p

BEFORE:
mister qualter as the apostle of the middle classes and we re glad twelcomed his gospe

AFTER:
mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
```

Notice that each of these examples are handpicked. Thus, the transcript resulting from the prefix beam search is also the true transcript.

### Dependencies
In order to run the examples with the language model you need to install the following Python packages:

* `numpy`
* `kenlm`

You find the `kenlm` package here: https://kheafield.com/code/kenlm/

You only need to install the Python wrapper from the `kenlm` package. This is done by following these steps:

1. Download the package: https://kheafield.com/code/kenlm.tar.gz
2. Unzip and `cd` to the main folder (this should contain a `setup.py`)
3. Install with pip: `pip install .`
