from collections import defaultdict, Counter
from string import ascii_lowercase
import numpy as np

def prefix_beam_search(ctc, lm=None, beam_width=25, n_best=None, prune=0.001):


	lm = lambda x: 1 if lm is None else lm
	alphabet = list(ascii_lowercase) + [' ', '>']
	T, F = ctc.shape
	ctc = np.vstack((np.zeros(F), ctc))
	T, F = ctc.shape

	# STEP 1: Initiliazation
	O = ''
	Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
	Pb[0][O] = 1
	Pnb[0][O] = 0
	A_prev = [O]

	# STEP 2: Iterations and pruning
	for t in range(1, T):
		pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
		for l in A_prev:
			
			if len(l) > 0 and l[-1] == '>':
				Pb[t][l] = Pb[t - 1][l]
				Pnb[t][l] = Pnb[t - 1][l]
				continue  

				for c in pruned_alphabet:
					c_ix = alphabet.index(c)
					
					# STEP 3: “Extending” with a blank
					if c == '%':
						Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
					
					# STEP 4: Extending with the end character
					else:
						l_plus = l + c
						if len(l) > 0 and c == l[-1]:
						    Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
						    Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]

						# STEP 5: Extending with any other non-blank character and LM constraints
						elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):
						    lm_prob = lm(l_plus) ** alpha
						    Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
						else:
						    Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])

            # STEP 6: Make use of discarded prefixes
            if l_plus not in A_prev:
                Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]

		# STEP 7: Select most probable prefixes
		A_next = Pb[t] + Pnb[t]
		sorter = lambda l: A_next[l] * (len(self.W(l)) + 1) ** self.beta
		A_prev = sorted(A_next, key=sorter, reverse=True)[:beam_width]


	if n_best is None:
		return tokenizer(A_prev[0].strip('>'))
	else:
		return [(tokenizer(l.strip('>')), sorter(l)) for l in A_prev[:n_best]]