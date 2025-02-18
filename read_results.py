import pickle
import matplotlib.pyplot as plt

with open('/home/ceron/xsbert/party_tokens_gruene_fdp.pkl', 'rb') as file:
    data = pickle.load(file)

from collections import Counter

# remove pairs where the tokens are the same
not_equal = []
for pair in data[('gruene', 'fdp')]:
    if pair[0]!=pair[1]:
        not_equal.append(pair)


# count the number of times each token pair appears
counter = Counter(not_equal)
# for pair in counter.most_common()[:-41:-1]:
for pair in counter.most_common(100):
    print(pair)

def plot_histogram(counter):
    # plot a histogram of the counts
    values = list(counter.values())
    bins = range(1, 10)
    hist, bin_edges, _ = plt.hist(values, bins=bins)

    plt.xlabel('Number of occurrences')
    plt.ylabel('Number of token pairs')

    # Annotate the bars with the bin counts
    for count, edge in zip(hist, bin_edges):
        plt.text(edge, count, str(int(count)), ha='left', va='bottom')

    plt.show()
    plt.savefig('histogram.png')



