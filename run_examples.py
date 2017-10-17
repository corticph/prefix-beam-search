from glob import glob
from string import ascii_lowercase
import pickle
import numpy as np
import kenlm
from prefix_beam_search import prefix_beam_search

class LanguageModel(object):
  """
  Loads a trained language model from an .arpa .file.

  After loader, the probability of the last word is returned given an input sentence.
  """

  def __init__(self, lm_path):
    self._model = kenlm.LanguageModel(lm_path)

  def __call__(self, sentence):
    scores = self._model.full_scores(sentence, bos=True, eos=False)
    return 10 ** list(scores)[-1][0]

def greedy_decoder(ctc):
  """
  Performs greedy decoding (max decoding) on the output of a CTC network.

  Args:
    ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)

  Retruns:
    string: The decoded CTC output.
  """

    alphabet = list(ascii_lowercase) + [' ', '>']
    alphabet_size = len(alphabet)

    #  collapse repeating characters
    arg_max = np.argmax(ctc, axis=1)
    repeat_filter = arg_max[1:] != arg_max[:-1]
    repeat_filter = np.concatenate([[True], repeat_filter])
    collapsed = arg_max[repeat_filter]

    # discard blank tokens (the blank is always last in the alphabet)
    blank_filter = np.where(collapsed < (alphabet_size - 1))[0]
    final_sequence = collapsed[blank_filter]
    full_decode = ''.join([alphabet[letter_idx] for letter_idx in final_sequence])

    return full_decode[:full_decode.find('>')]

if __name__ == '__main__':

    #lm = LanguageModel('lm.binary')
    lm = LanguageModel('nlm.arpa')
    for example_file in glob('examples/*.p'):
        example = pickle.load(open(example_file, 'rb'))
        before_lm = greedy_decoder(example)
        after_lm = prefix_beam_search(example, lm=lm)
        print('\n{}'.format(example_file))
        print('\nBEFORE:\n{}'.format(before_lm))
        print('\nAFTER:\n{}'.format(after_lm))