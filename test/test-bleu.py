from nltk.translate.bleu_score import corpus_bleu

pre1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
        'ensures', 'that', 'the', 'military', 'always',
        'obeys', 'the', 'commands', 'of', 'the', 'party']

ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
         'ensures', 'that', 'the', 'military', 'always',
         'obeys', 'the', 'commands', 'of', 'the', 'party','aaa','aaa','aaa']

pre2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
        'interested', 'in', 'world', 'history']
ref2a = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
         'interested', 'in', 'world', 'history','aaa']

list_of_references = [[ref1a], [ref2a]]
predictions = [pre1, pre2]
print(corpus_bleu(list_of_references, predictions))
