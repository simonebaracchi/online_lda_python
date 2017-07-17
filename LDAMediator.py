#!/usr/bin/python

# Simplified interface to LDA.

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb
import time
import sys

from printer import Printer
import argparse

class LDAMediator:
    def __init__(self):
        # The total number of documents in the corpus.
        self.D = 1024
        # The number of topics.
        self.K = 5
        # Every word not in vocab will be ignored, so we must keep track of them
        empty_vocab = set()
        # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
        self.alpha = 1./self.K  # prior on topic weights theta
        self.eta   = 1./self.K  # prior on p(w|topic) Beta
        self.tau_0 = 1024  # learning parameter to downweight early documents
        self.kappa = 0.7  # learning parameter; decay factor for influence of batches
        self.olda = onlineldavb.OnlineLDA(empty_vocab, self.K, self.D, self.alpha, 1./self.K, self.tau_0, self.kappa)

    def load(self, docset=None):
        if docset == None:
            docset = []
        # update dictionary
        self.olda.merge_vocab(docset)

        # The number of documents to analyze each iteration.
        batchsize = len(docset)
        # The total number of documents in the corpus.
        self.D += batchsize

        # Give them to online LDA
        (gamma, bound) = self.olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, self.olda._vocab)

        numpy.savetxt('lambda-final.dat', self.olda._lambda)
        numpy.savetxt('gamma-final.dat', gamma)

