from datasets import load_dataset
from utils.word2vec import NaiveCBOW, SkipGram, createDictionaryAndCounter, createCorpus, textToWords
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Testing word2vec implementation")
    parser.add_argument("--is_CBOW", action="store_true", help="Wether to use CBOW. Without it skip-gram will be tested.")
    parser.add_argument("--logdir", type=str, default="out", help="Filepath to log directory.")
    parser.add_argument("--d", type=int, default=200, help="Hidden dimension in word2vec architecture.")
    parser.add_argument("--dataset_name", "--dn", default="imdb", type=str, help="Name of dataset.")
    parser.add_argument("--reduce_dataset_size", "--r", action="store_true", help="Wether to reduce dataset size.")
    parser.add_argument("--dataset_size", "--ds", default=500, type=int, help="If reduce dataset,\
                         then this parameter defines number of samples to take from it.")
    parser.add_argument("--min_occurences", type=int, help="Minimal occurences for word to be present in dictionary.", required=True)
    parser.add_argument("--max_occurences", default=1000, type=int, help="Maximal occurences for word to be present in dictionary.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    print(train_df.head())

    X_train = train_df["text"].values
    X_test = test_df["text"].values
    # Different column is "label" in imbdb dataset.
    if args.reduce_dataset_size:
        X_train = X_train[:args.dataset_size]
        X_test = X_test[:args.dataset_size]
    dictionary, counter = createDictionaryAndCounter(X_train, min_value=args.min_occurences, max_value=args.max_occurences)
    D = len(dictionary) # For min_value=5 number of keys is around 4000.
    print(f"Number of keys: {D}")
    
    if args.is_CBOW:
        cbow = NaiveCBOW(len(dictionary), args.d, dictionary, np.array([-4,-3,-2,-1,1,2,3,4]))
        cbow.train(X_train, X_test, 5, 50, 0.01,  computeUsingBatch=True)
    else:
        sg = SkipGram(len(dictionary), args.d, counter, window=10, k=10, seed=42)
        sg.train(
            createCorpus(X_train, dictionary=dictionary),
            createCorpus(X_test, dictionary=dictionary),
            5,
            1000,
            0.1,
            100,
            logdir=(args.logdir + "/skip_gram")
        )

if __name__ == "__main__":
    main()