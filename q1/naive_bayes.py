import numpy as np

class NaiveBayes:
    def __init__(self):
        self.smoothening = 1
        self.phi_y = [] # initialize it later to be a list of size num_classes and phi_y[i] is P(Y=i)
        self.phi_j_y = {} # initialize it later to be a dictionary of size num_classes and phi_j_y[i] is a dictionary of P(X=j|Y=i)
        self.num_classes = 0
        self.vocab = set() # set of all unique words in the training data #can always find the length of vocab by len(vocab)
        self.total_words_in_class = [] # initialize it later to be a list of size num_classes and total_words_in_class[i] is the total number of words in class i
        pass
        
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        num_of_classes = df[class_col].nunique()
        self.num_classes = num_of_classes
        self.smoothening = smoothening
        self.phi_y = [0 for _ in range(num_of_classes + 1)]
        self.phi_j_y = {i: {} for i in range(num_of_classes + 1)}
        self.total_words_in_class = [0 for _ in range(num_of_classes + 1)]
        class_freq = df[class_col].value_counts().to_dict()
        class_word_freq = {i: {} for i in range(num_of_classes + 1)}
        #iterating thru each data point of x_i and y_i
        for index, row in df.iterrows():
            y_i = row[class_col]
            x_i = row[text_col]
            self.total_words_in_class[y_i] += len(x_i)   #assuming classes are 1-indexed
            curr_class_word_freq = class_word_freq[y_i]
            for word in x_i:
                self.vocab.add(word)
                curr_class_word_freq[word] = curr_class_word_freq.get(word, 0) + 1
        vocab_size = len(self.vocab)
        m = sum(class_freq.values())
        # print(m)
        
        #compute phi_y and phi_j_y
        for c in range(num_of_classes): #assuming classes are 0-indexed
            # print(class_freq.get(c, 0), m)
            self.phi_y[c] = np.log(class_freq.get(c, 0) / m)
            curr_class_word_freq = class_word_freq[c]
            total_words = self.total_words_in_class[c]
            for word, count in curr_class_word_freq.items():
                self.phi_j_y[c][word] = np.log((count + smoothening) / (total_words + smoothening * vocab_size))
            #for words not in the class, we can set their log-probability to be
            self.phi_j_y[c]['<UNK>'] = np.log((smoothening) / (total_words + smoothening * vocab_size))
    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        #p(y|x) = p(x, y) / p(x) which is proportional to 
        #         p(x|y) * p(y) = p(y) * product of p(x_j|y) for each word x_j in x
        num_of_classes = self.num_classes
        df[predicted_col] = -1
        for index, row in df.iterrows():
            x_i = row[text_col]
            #calculate log-probabilities for each class and then take argmax
            log_probs = [float('-inf') for _ in range(num_of_classes + 1)]
            for c in range(num_of_classes): #assuming classes are 1-indexed
                log_prob = self.phi_y[c]
                for word in x_i:
                    if word in self.phi_j_y[c]:
                        log_prob += self.phi_j_y[c][word]
                    else:
                        log_prob += self.phi_j_y[c]['<UNK>']
                log_probs[c] = log_prob
            predicted_class = np.argmax(log_probs)
            df.at[index, predicted_col] = predicted_class
        return df
    