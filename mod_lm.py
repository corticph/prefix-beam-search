import progressbar
import time

with open('vocab.txt', 'r') as vfile:
    vocab = tuple([w.strip() for w in vfile.read().split('\n')])

print('LOADING START')
n_lines = 1000 + 191738 + 38049785 #+ 203284146 + 434430781
count = 0
bar = progressbar.ProgressBar(max_value=n_lines)
with open('/nas/users/labo/libri_models/word_lm/lm_word_4_fulldata_vocab200K/lm.arpa', 'r') as ifile:
    with open('nlm.arpa', 'a') as ofile:
        for line in ifile:
            count += 1
            bar.update(count)
            line = line.strip('\n')
            tabs = line.split('\t')
            if line == '\3-grams:':
                break

            elif len(tabs) >= 2:
                words = tabs[1].split()
                for w in words:
                    if w in vocab:
                        print(line, file=ofile)
                        break

            else:
                print(line, file=ofile)


"""
print('LOADING DONE', time())




with open('nlm.arpa', 'a') as lfile:
    bar = progressbar.ProgressBar()
    lines = arpa.split('\n')
    for l in bar(lines):
        tabs = l.split('\t')
        if len(tabs) >= 3:
            for w in tabs[1:-1]:
                if w in vocab:
                    print(l, file=ofile)
                    break

        else:
            print(l, file=ofile)

"""

