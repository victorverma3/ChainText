from collections import defaultdict
from nltk.tokenize import word_tokenize
import numpy as np
import os
import string
import time


class MarkovChain:
    def __init__(self, memory, sources=[], texts=[]):

        self.memory = memory
        self.sources = sources
        self.texts = texts
        self.words = []
        self.unique_words = []
        self.next_words = {}
        self.sentence_starters = []
        self.word_to_index = {}
        self.transition_matrix = []

    def __repr__(self):

        # prints the relevant Markov chain information
        return f"\nMarkov chain properties:\ntrained on {len(self.texts)} piece[s] of text\ntrained on {len(self.unique_words)} unique words\norder of memory is {self.memory}"

    def initialize(self):

        start_initialize = time.perf_counter()

        # initializes the Markov chain parameters
        self.parse()
        self.get_unique_words()
        self.get_next_words_and_sentence_starters()
        self.build_markov_chain()

        done_initialize = time.perf_counter()
        print(
            f"initialized Markov chain in {done_initialize - start_initialize} seconds"
        )

    def add_text_as_string(self, text):

        # verifies parameters
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        # adds the string to the list of training texts for the Markov chain
        self.texts.append(text)

        # reinitializes the Markov chain object to update the parameters
        self.initialize()

    def add_texts_from_string_list(self, texts):

        # verifies parameters
        if not isinstance(texts, list) and not all(
            isinstance(item, str) for item in texts
        ):
            raise ValueError("texts must be a list of strings")

        # adds the list of strings to the list of training texts for the Markov chain
        self.texts.extend(texts)

        # reinitializes the Markov chain object to update the parameters
        self.initialize()

    def add_text_from_file(self, file):

        # adds the text from the file to the list of training texts
        try:
            if file not in self.sources:
                with open(file, "r") as f:
                    self.sources.append(file)
                    self.texts.append(f.read())
        except:
            raise Exception("failed to read file")

        # reinitializes the Markov chain object to update the parameters
        self.initialize()

    def add_texts_from_file_list(self, files):

        # adds the text from the files to the list of training texts
        try:
            for file in files:
                if file in self.sources:
                    continue
                else:
                    with open(file, "r") as f:
                        self.sources.append(file)
                        self.texts.append(f.read())
        except:
            raise Exception("failed to read file list")

        # reinitializes the Markov chain object to update the parameters
        self.initialize()

    def parse(self):

        if self.memory == 1:
            # parses the training text into a list of all of the words and punctuation
            words = []
            for index, text in enumerate(self.texts):
                if index == 0:
                    words.extend(word_tokenize(text.lower()))
                else:
                    words.extend(["BREAK"] + word_tokenize(text.lower()))

        self.words = words

    def get_unique_words(self):

        # creates a set of all of the unique words in the training text
        self.unique_words = set(self.words)
        self.unique_words.discard("BREAK")

    def get_next_words_and_sentence_starters(self):

        if self.memory == 1:
            # initializes the dictionary storing the next-word options
            next_words = {}
            for word in self.unique_words:
                next_words[word] = []

            # initializes the list of sentence-starting words
            sentence_starters = [self.words[0]]

            # updates the next_words dictionary with the next-word options
            for i in range(len(self.words) - 1):

                # only supports sentences ending in period, exclamation, or question
                if self.words[i] in [".!?"]:
                    sentence_starters.append(self.words[i + 1])
                elif self.words[i] == "BREAK":
                    sentence_starters.append(self.words[i + 1])
                    continue

                # skips the break between pieces of text
                if self.words[i + 1] != "BREAK":
                    next_words[self.words[i]].append(self.words[i + 1])

        self.next_words = next_words
        self.sentence_starters = sentence_starters

    def build_markov_chain(self):

        if self.memory == 1:

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

        self.word_to_index = word_to_index
        self.transition_matrix = transition_matrix

    def generate_next_word(self, current_word, memory):

        # verifies parameters
        if memory not in [1, 2]:
            raise ValueError("memory must be the integer 1 or 2")

        if memory == 1:
            # determines the row of the transition matrix to use
            index = self.word_to_index[current_word]

            # chooses the next word from the sentence starters if the sentence is over (only supports sentences ending in period, exclamation, and question)
            if current_word in ".!?":
                next_word = np.random.choice(self.sentence_starters)

            # chooses the next word based on the transition matrix
            else:
                next_word = np.random.choice(
                    list(self.word_to_index.keys()),
                    p=self.transition_matrix[index],
                )
            return next_word

    def generate(self, length, memory):

        start_generation = time.perf_counter()

        # chooses the first word of the generation
        current_word = np.random.choice(self.sentence_starters)
        generation = f"\n{current_word.title()}"
        curr = 1

        # continues adding words until the limit is reached
        while curr < length:
            next_word = self.generate_next_word(current_word, memory)

            # only adds spaces when the next word isn't punctuation
            if next_word not in string.punctuation:
                generation += " "

            # appends the next word to the generated text
            if current_word in ".!?":
                generation += f"{next_word.title()}"
            elif next_word == "i":
                generation += f"{next_word.upper()}"
            else:
                generation += next_word

            # updates current word
            current_word = next_word
            curr += 1

        done_generation = time.perf_counter()
        print(f"generated text in {done_generation - start_generation} seconds")
        return generation


def generate_text_from_source(length=200, memory=1):

    # verifies parameters
    if memory not in [1, 2]:
        raise ValueError("memory must be the integer 1 or 2")

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

    # reads all .txt files in the source directory
    start_reading = time.perf_counter()
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
    print(f"\nread source files in {done_reading - start_reading} seconds")

    # creates the Markov chain object
    chain = MarkovChain(memory, sources, texts)
    chain.initialize()

    # generates the text
    print(chain.generate(length, memory))

    # prints the properties of the Markov chain
    print(chain)


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

        # gets the user-input number for memory considered in Markov chain
        memory = int(
            input(
                "how many states should be considered in memory? (integer 1 or 2) (-1 to quit) "
            )
        )
        if memory == -1:
            quit = True
            break
        elif memory not in [1, 2]:
            raise ValueError("memory must be the integer 1 or 2")

        # generates text
        generate_text_from_source(length, memory)
        break
