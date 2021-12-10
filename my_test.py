# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:59:15 2021

@author: Patrick
"""

from utils import vec_fun, split_data, my_rf, tokenize, my_nb
from utils import perf_metrics, open_pickle, my_pca
import pandas as pd

if __name__ == "__main__":

    base_path = "/Users/jean-sebastienprovost/Desktop/Jobs/Nobul/coding_test/"
              
    final_data = open_pickle(base_path, "data.pkl")

   # Preprocessing steps added
    tokens = [tokenize(x) for x in final_data.body_basic]
    tokens = [' '.join(x) for x in tokens]

    # CountVectorizer with the newly generated tokens
    my_vec_text = vec_fun(tokens, base_path)

    # Train test split
    X_train, X_test, y_train, y_test = split_data(
        my_vec_text, final_data.label, 0.1)

    # Random Forest and Naive Bayes Models with the tokens
    rf_model = my_rf(
        X_train, y_train, base_path)
    rf_metrics = perf_metrics(rf_model, X_test, y_test)
    nb_model = my_nb(
        X_train, y_train, base_path)
    nb_metrics = perf_metrics(nb_model, X_test, y_test)

    # Performed PCA with the original tokens
    pca_data = my_pca(my_vec_text, 0.95, base_path)

    # Train test split with the PCA data set
    X_train, X_test, y_train, y_test = split_data(
        pca_data, final_data.label, 0.1)

    # Random Forest and Naive Bayes Models with the tokens after PCA
    rf_pca_model = my_rf(
        X_train, y_train, base_path)
    rf_pca_metrics = perf_metrics(rf_pca_model, X_test, y_test)
    nb_pca_model = my_nb(
        X_train, y_train, base_path)
    nb_pca_metrics = perf_metrics(nb_pca_model, X_test, y_test)

    
