import argparse 
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import time

class DecisionNode:
    def __init__(self,feature=None,split=None,height=None): # split chosen from x values
        self.feature = feature
        self.split = split
        self.left = None
        self.right = None 
        self.height = 0

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit 
        self.n_nodes=0

    def fit(self, X:np.ndarray, improved=False, e=0):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        # get number of attributes 
        attributes = X.shape[1]
        if e >=self.height_limit or len(X) <= 1:
            self.n_nodes+=1
            return X
        else:
            # get q and p 
            if improved: 
                splits = []
                for i in range(2):
                    q = np.random.randint(attributes)
                    p = np.random.uniform(min(X[:,q]),max(X[:,q]))
                    left = X[(X[:,q]<=p)]
                    right = X[(X[:,q]>p)]
                    splits.append([min(len(left), len(right)), q, p,left,right])
                splits.sort(key=lambda x: x[0])
                q = splits[0][1]
                p = splits[0][2]
                left = splits[0][3]
                right = splits[0][4]
            else: 
                q = np.random.randint(attributes)
                p = np.random.uniform(min(X[:,q]),max(X[:,q]))
                left = X[(X[:,q]<=p)]
                right = X[(X[:,q]>p)]
                
                
            # build decision node 
            root = DecisionNode(feature=q, split=p)
            self.n_nodes+=1
            root.height = e + 1
            # build left and right splits 
            root.left = self.fit(X=left,e=root.height,improved=improved)
#             if isinstance(root.left, DecisionNode):
#                 self.n_nodes+=1
            root.right = self.fit(X=right,e=root.height,improved=improved)
#             if isinstance(root.right, DecisionNode):
#                 self.n_nodes+=1
            self.root=root
        return self.root
    
class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.trees = [] 
        self.height_limit = int(np.ceil(np.log2(sample_size)))
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.root = None 
        self.scores = None
        self.predictions = None 
        self.avg_lengths = None 
        
    def c_sample_size(self,n):
        if n == 2:
            return 1 
        elif n > 2:
            c = 2*(np.log(n-1)+0.5772156649) - (2*(n-1)/n)
            return c
        else:
            return 0

    def compute_path_length(self,X,tree,e=0):
        if isinstance(tree, np.ndarray):
            length = e + self.c_sample_size(len(tree))
            self.avg_lengths[(X[:,-1].astype(int))]+=length
        else:
            attribute = tree.feature
            split = tree.split
#         if row[attribute]<split:
            self.compute_path_length(X[(X[:,attribute]<=split)], tree=tree.left, e=e+1)
#         else:
            self.compute_path_length(X[(X[:,attribute]>split)], tree=tree.right, e=e+1)
        
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        num_rows = X.shape[0]
        for i in range(self.n_trees):
            sample_idx = np.random.randint(low=0,high=num_rows,size=self.sample_size)
            itree = IsolationTree(self.height_limit)
            itree.fit(X[(sample_idx)], improved)
            self.trees.append(itree)
        return self
    
    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """   
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.avg_lengths = np.zeros((len(X)))
        X = np.append(X, np.arange(len(X)).reshape(-1,1), axis=1)
        
#         length_matrix = np.zeros(shape=(len(X),len(self.trees)))
#         for row in range(len(X)):
#             for itree in range(len(self.trees)):
#                 length_matrix[row,itree]=(self.compute_path_length(X[row],self.trees[itree].root))
#         self.avg_lengths = np.mean(length_matrix,axis=1)
#         return self.avg_lengths 

        for itree in self.trees:
            self.compute_path_length(X,itree.root,e=0)
        self.avg_lengths = self.avg_lengths/len(self.trees)
        return self.avg_lengths

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.scores = 2**(-1*self.path_length(X)/self.c_sample_size(self.sample_size))
        return self.scores
        
    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        self.predictions = np.array([1 if score >= threshold else 0 for score in scores])
        return self.predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.predict_from_anomaly_scores(scores = self.anomaly_score(X), threshold=threshold)

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    if isinstance(y, pd.DataFrame):
            y = y.values

    y_score = list(zip(scores,y))
    acceptable_thresholds = []
   
    for threshold in np.arange(1,0,step=-0.01):
        fn = 0
        fp = 0
        tp = 0 
        tn = 0
        for score, y_i in y_score:
            if (score > threshold) and (y_i == 1):
                tp += 1
            elif (score > threshold) and (y_i == 0):
                fp += 1 
            elif (score < threshold) and (y_i == 1):
                fn += 1
            else:
                tn += 1
        try:
            if (tp/(tp+fn)) > desired_TPR:
                FPR = fp/(fp+tn)
                acceptable_thresholds.append((threshold,FPR))
        except:
            pass
    acceptable_thresholds.sort(key=lambda x: x[1])
    assert len(acceptable_thresholds)>0, "Desired TPR not met."

    return acceptable_thresholds[0][0], acceptable_thresholds[0][1]



if __name__ == '__main__':

    # Setting up command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="available modes: train or predict")
    parser.add_argument("X", help="X csv file")
    parser.add_argument("-y","-Y", help="Y csv file, required when mode is train", metavar='')
    parser.add_argument("-t","--trees", help="number of trees to use when building iforest, default is 300",\
                        type=int, default=300, metavar='')
    parser.add_argument("--tpr", help="desired TPR to select probability threshold when calculating FPR, default is 0.80",\
                        type=float, default=0.80, metavar='')
    parser.add_argument("-s","--sample_size", help="number of samples to use when training an itree, default is 256",\
                        type=int, default=256, metavar='')
    parser.add_argument("-i", "--improved", action="store_true", help="include flag to train with improved version. \
                        user will experience slightly slower fit times")
    parser.add_argument("-n","--name", help="filename for trained model, default is fitted_iForest.pkl",\
                        default='fitted_iForest.pkl', metavar='')
    args = parser.parse_args()

    # Check for valid mode 
    assert args.mode in ["train", "predict"], "Mode needs to be either 'train' or 'predict'."

    # Training 
    if args.mode == "train":

        if args.y is None:
            parser.error('Y is required when training')

        X = pd.read_csv(os.path.join("data",args.X))
        Y = pd.read_csv(os.path.join("data",args.y))

        itree = IsolationTreeEnsemble(sample_size=args.sample_size, n_trees=args.trees)

        # fit itree model 
        fit_start = time.time()
        itree.fit(X, improved=args.improved)
        fit_stop = time.time()
        fit_time = fit_stop - fit_start
        print(f"INFO fit time {fit_time:3.2f}s")

        n_nodes = sum([t.n_nodes for t in itree.trees])
        print(f"INFO {n_nodes} total nodes in {args.trees} trees")

        # get score on training set 
        score_start = time.time()
        scores = itree.anomaly_score(X)
        score_stop = time.time()
        score_time = score_stop - score_start
        print(f"INFO score time {score_time:3.2f}s")

        # find FPR and best threshold 
        threshold, FPR = find_TPR_threshold(Y, scores, args.tpr)
        y_pred = itree.predict_from_anomaly_scores(scores, threshold=threshold)

        confusion = confusion_matrix(Y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        print(f"{args.trees} trees at desired TPR {args.tpr*100.0:.1f}% getting FPR {FPR:.4f}%")
        print(f"Saving trained model to trained_model directory...")

        # save model and threshold value 
        with open (os.path.join('trained_model',args.name), 'wb') as f:
            pickle.dump(itree, f, pickle.HIGHEST_PROTOCOL)
        threshold_path = os.path.join('trained_model',args.name.split('.')[0]+'_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(str(threshold))

    # Prediction 
    if args.mode=="predict":

        if args.y is not None:
            print('Warning: Y is not needed in predict and will be ignored')

        # check model has been trained, if so load and model, threshold, and data
        assert os.path.isfile(os.path.join('trained_model',args.name)), 'no saved model. iForest must be trained first'
        with open(os.path.join('trained_model',args.name), "rb") as f:
            itree = pickle.load(f)
        threshold_file = args.name.split('.')[0]+'_threshold.txt'
        with open(os.path.join('trained_model',threshold_file), "r") as f:
            threshold = float(f.read())
        X = pd.read_csv(os.path.join("data",args.X))

        # calculating anomaly scores 
        scores = itree.anomaly_score(X)

        # calculating hard predictions based on threshold values 
        y_pred = itree.predict_from_anomaly_scores(scores, threshold=threshold)

        # saving predictions
        prediction_path= os.path.join('trained_model',args.name.split('.')[0]+'_predictions.csv')
        print(f"Saving predictions to {prediction_path}...")
        predictions = pd.DataFrame({'index':np.arange(len(scores)),'scores':scores,'prediction':y_pred})
        predictions.to_csv(prediction_path, index=False)



















