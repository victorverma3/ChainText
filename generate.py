from collections import defaultdict
from nltk.tokenize import word_tokenize
import numpy as np
import os
import string
import time


class MarkovChain:
    def __init__(self, sources, texts, transition_states):
        self.sources = sources
        self.texts = self.verify_texts(texts)
        self.words = self.parse()
        self.unique_words = self.get_unique_words()
        self.next_words, self.sentence_starters = self.create_next_words()
        self.word_to_index, self.transition_matrix = self.build_markov_chain()
        self.transition_states = transition_states
        self.transition_matrices = self.build_transition_matrices()

    def __repr__(self):
        # prints the transition matrix
        return repr(self.transition_matrix)

    def verify_texts(self, texts):
        # verifies that texts is a list
        if type(texts) != list:
            raise ValueError("texts must be passed in as elements in a list")
        return texts

    def parse(self):
        # parses the training text into a list of all of the words and punctuation
        words = []
        for text in self.texts:
            words.extend(word_tokenize(text))
        return words

    def get_unique_words(self):
        # creates a set of all of the unique words in the training text
        return set(self.words)

    def add_text(self, texts):
        self.verify_texts(texts)
        # updates markov chain to include a new source text
        self.texts.extend(texts)
        self.words = self.parse()
        self.unique_words = self.get_unique_words()
        self.next_words, self.sentence_starters = self.create_next_words()
        self.word_to_index, self.transition_matrix = self.build_markov_chain()

    def create_next_words(self):
        # initializes the dictionary storing the next-word options
        next_words = {}
        for word in self.unique_words:
            next_words[word] = []

        # initializes the list of sentence-starting words
        sentence_starters = [self.words[0]]

        # updates the next_words dictionary with the next-word options
        for i in range(len(self.words) - 1):
            next_words[self.words[i]].append(self.words[i + 1])
            # only supports sentences ending in period, exclamation, or question
            if self.words[i] in [".!?"]:
                sentence_starters.append(self.words[i + 1])
        return next_words, sentence_starters

    def build_markov_chain(self):
        #  maps the words to their indices in the transition matrix
        word_to_index = {word: i for i, word in enumerate(self.unique_words)}

        # initializes the transition matrix
        num_words = len(self.unique_words)
        transition_matrix = np.zeros((num_words, num_words))

        # iterates through the words
        for word, next_words in self.next_words.items():
            current_word_index = word_to_index[word]
            total_transitions = len(next_words)
            next_word_counts = defaultdict(int)

            # counts the occurences of the next-word
            for next_word in next_words:
                next_word_counts[next_word] += 1

            # calculates the probability of the next-word given the current word
            for next_word, count in next_word_counts.items():
                next_word_index = word_to_index[next_word]

                # updates the transition matrix
                transition_matrix[current_word_index][next_word_index] = (
                    count / total_transitions
                )
        return word_to_index, transition_matrix

    def build_transition_matrices(self):
        transition_matrices = []

        # creates i + 1 state transition matrices
        for i in range(self.transition_states):
            transition_matrices.append(
                np.linalg.matrix_power(self.transition_matrix, i + 1)
            )
        return transition_matrices

    def generate_next_word(self, current_word):
        # determines the row of the transition matrix to use
        index = self.word_to_index[current_word]

        # chooses the next word from the sentence starters if the sentence is over (only supports sentences ending in period, exclamation, and question)
        if current_word in [".!?"]:
            next_word = np.random.choice(self.sentence_starters)

        # chooses the next word based on the transition matrix
        else:
            next_word = np.random.choice(
                list(self.word_to_index.keys()), p=self.transition_matrix[index]
            )
        return next_word

    def generate(self, length, transition_states):
        # chooses the first word of the generation
        current_word = np.random.choice(self.sentence_starters)
        generation = f"\n{current_word}"
        curr = 1

        # continues adding words until the limit is reached
        while curr < length:
            next_word = self.generate_next_word(current_word)

            # only adds spaces when the next word isn't punctuation
            if next_word not in string.punctuation:
                generation += " "

            # appends the next word to the generated text
            generation += next_word
            current_word = next_word
            curr += 1
        generation += "\n"
        return generation


def generate_text_from_source(length=200, transition_states=1):

    if transition_states not in [1, 2]:
        raise ValueError("number of transition states must be the integer 1 or 2")
    # gets the user-input source dataset
    source = input("choose a source dataset: ")
    if source not in ["all", "nursery", "philosophy", "poems", "trump", "victor"]:
        raise ValueError("invalid source dataset")
    sourceMap = {
        "all": "./sources",
        "nursery": "./sources/nursery",
        "philosophy": "./sources/philosophy",
        "poems": "./sources/poems",
        "trump": "./sources/trump",
        "victor": "./sources/victor",
    }
    start_reading = time.perf_counter()

    # reads all .txt files in the source directory
    try:
        sources = []
        texts = []
        for root, dirs, files in os.walk(f"{sourceMap[source]}"):
            for file in files:
                if file.endswith(".txt"):
                    sources.append(file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        texts.append(f.read())
    except:
        raise ValueError("error processing poems - try again or try a different input")
    done_reading = time.perf_counter()
    print(f"read source files in {done_reading - start_reading} seconds")

    # creates the MarkovChain object
    chain = MarkovChain(sources, texts, transition_states)
    done_markov_chain = time.perf_counter()
    print(f"created Markov chain in {done_markov_chain - done_reading} seconds")

    # generates the text
    print(chain.generate(length, transition_states))
    done_generation = time.perf_counter()
    print(f"generated text in {done_generation - done_markov_chain} seconds\n")


if __name__ == "__main__":
    while True:

        # gets the user-input generation length
        length = int(
            input("\nhow many words should be generated? (integer > 0) (-1 to quit) ")
        )
        if length == -1:
            break
        elif length < 1:
            raise ValueError("length must be an integer greater than or equal to 1")

        # gets the user-input number of transition states
        transition_states = int(
            input(
                "how many transition states should be considered? (integer 1 or 2) (-1 to quit) "
            )
        )
        if transition_states == -1:
            quit = True
            break
        elif transition_states not in [1, 2]:
            raise ValueError("number of transition states must be the integer 1 or 2")

        # generates text
        generate_text_from_source(length, transition_states)
        break
