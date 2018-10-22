from __future__ import print_function

import six.moves.cPickle as pickle
import matplotlib.pyplot as plt


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams




class dA(object):

    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None,

    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:

            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
            initial_momemtum = numpy.asarray(
                numpy.zeros((n_visible, n_hidden)),
                dtype=theano.config.floatX
            )
            momemtum = theano.shared(value=initial_momemtum, name='momemtum', borrow=True)
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            momemtum_bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
            momemtum_bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b_momemtum',
                borrow=True
            )

        self.W = W
        self.momentum = momemtum
        # b corresponds to the bias of the hidden
        self.b = bhid
        self.b_momemtum = momemtum_bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        self.b_prime_momemtum = momemtum_bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        self.mom = [self.momentum, self.b_momemtum, self.b_prime_momemtum]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self,learning_rate, beta=0.9):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        max_x = T.max(self.x)
        min_x = T.min(self.x)
        self.x = (self.x - min_x) / (max_x - min_x)
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        # L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L_prime = T.sum(T.square(z - self.x), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L_prime)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for momen, gparm1 in zip(self.mom, gparams):
            updates.append((momen, momen * beta + (1 - beta) * gparm1))
        for param, momen in zip(self.params, self.mom):
            updates.append((param, param - learning_rate * momen))

        return (cost, updates)


train_costs = []
valid_costs = []
epoches = []

def train_Ae(autoencoder,datasets,targ_class=None, learning_rate=0.1, training_epochs=100,
             dataset='mnist.pkl.gz',
             batch_size=20, output_folder='dA_data'):


    ####################################
    # LOADING DATA  #
    ####################################
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # TRAINING THE MODEL #
    ####################################

    cost, updates = autoencoder.get_cost_updates(
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        v = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        for batch_valid_index in range(n_valid_batches):
            v.append(validate_model(batch_valid_index))
        epoches.append(epoch)
        train_costs.append(numpy.mean(c, dtype='float64'))
        valid_costs.append(numpy.mean(v, dtype='float64'))
        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))
        print('Validate epoch %d, cost ' % epoch, numpy.mean(v, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)
    plt.plot(epoches, train_costs, 'b-', label="Training")
    plt.plot(epoches, valid_costs, 'r-', label="Validation")
    plt.xlabel("Ecpoh")
    plt.ylabel("Error")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    print(('The  code f ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)

    # save the best model
    with open('bestminst_autoEncoder' + str(targ_class) + '.pkl', 'wb') as f:
        pickle.dump(autoencoder, f)

    os.chdir('../')

