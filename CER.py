def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]

texts = []
with open("/media/qw/data/Experiment/Encoder_selfAtt/test/HarryPotter.txt", "r") as f:
    for line in f:
        line = line.strip()
        if len(line):
            texts.append(line)
with open("/media/qw/data/Experiment/Encoder_selfAtt/test/2.txt", "r") as f:
    for line in f:
        line = line.strip()
        if len(line):
            texts.append(line)

test = texts[0]
stt = texts[1]

k = levenshtein(stt, test)
cer = (k/len(test))*100
# print(len(test), k, cer)

from scipy.special import beta, binom
n = 11
for k in range(n):
    b = binom(n,k)
    p = beta(k+0.1, n-k+0.9) / beta(0.1,0.9)
    print(b*p)