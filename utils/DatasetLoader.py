
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import numpy
import theano
import theano.tensor as T


def load_minst_data(dataset, label=None):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        if label is not None:
            trueIndexes = numpy.where(data_y == label)
            data_x = data_x[trueIndexes]
            data_y = data_y[trueIndexes]

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_kdd_data(dataset, test_dataset, target_class="normal", classification=None):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the train dataset
    :type test_dataset: string
    :param dataset: the path to the dataset (here KDD)
    '''

    d = defaultdict(LabelEncoder)

    def labels_to_classes(set):
        ''' Transforms KDD dataset attack labels into the for main attack n_classes
        as described in the paper
        'A Study on NSL-KDD Dataset for Intrusion
        Detection System Based on Classification
        Algorithms'
        https://pdfs.semanticscholar.org/1b34/80021c4ab0f632efa99e01a9b073903c5554.pdf
        :type set: dataframe
        :param set: the KDD dataset with the column at index 41 being the label.
        '''

        def to_list(string):
            return [x.strip() for x in string.split(',')]

        probe = "portsweep, ipsweep, queso, satan, msscan, ntinfoscan, lsdomain, illegal-sniffer, mscan, saint, nmap"
        probe = to_list(probe)
        dos = "apache2, smurf, neptune, dosnuke, land, pod, back, teardrop, tcpreset, syslogd, crashiis, arppoison, mailbomb, selfping, processtable, udpstorm, warezclient, worm"
        dos = to_list(dos)
        r2l = "dict, netcat, sendmail, imap, ncftp, xlock, xsnoop, sshtrojan, framespoof, ppmacro, guest, netbus, snmpget, ftpwrite, httptunnel, phf, named, snmpgetattack, snmpguess, guess_passwd, spy, ftp_write, multihop, warezmaster"
        r2l = to_list(r2l)
        u2r = "sechole, xterm,httptunnel, eject, ps, nukepw, secret, perl, yaga, fdformat, ffbconfig, casesen, ntfsdos, ppmacro, loadmodule, sqlattack, buffer_overflow, rootkit"
        u2r = to_list(u2r)
        for c in probe:
            set.ix[set[41] == c + '.', 41] = "probe"
        for c in dos:
            set.ix[set[41] == c + '.', 41] = "dos"
        for c in r2l:
            set.ix[set[41] == c + '.', 41] = "r2l"
        for c in u2r:
            set.ix[set[41] == c + '.', 41] = "u2r"
        set.ix[set[41] == "normal.", 41] = "normal"

        return set

    def getTuple(set):
        ''' Returns a tuple of numpy arrays form (input, target)
        :type set: pandas dataframe
        :param set: the KDD dataset loaded using pandas
        '''
        return set.ix[:, 0:40].values, set.iloc[:, 41].values

    def fit_transform(set):
        ''' Fits to the set provided and returns an encoded dataset. Use on training data.
        :type set: dataframe
        :param set: the KDD dataset loaded using pandas
        '''
        # encode columns 1, 2, 3 and the target
        set1 = set.ix[:, (1, 2, 3, 41)].apply(lambda x: d[x.name].fit_transform(x))

        return set1.combine_first(set)

    def transform(set):
        ''' Returns an encoded dataset. Use on new/test data.
        :type set: dataframe
        :param set: the KDD dataset loaded using pandas
        '''
        # encode columns 1, 2, 3 and the target
        set1 = set.ix[:, (1, 2, 3, 41)].apply(lambda x: d[x.name].transform(x))

        return set1.combine_first(set)

    def inverse_transform(set):
        ''' Returns a decoded dataset.
        :type set: dataframe
        :param set: the encoded KDD dataset
        '''
        # encode columns 1, 2, 3 and the target
        set1 = set.ix[:, (1, 2, 3, 41)].apply(lambda x: d[x.name].inverse_transform(x))

        return set1.combine_first(set)

    #############
    # LOAD DATA #
    #############

    dataset = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        dataset
    )
    test_dataset = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        test_dataset
    )

    # Load the training dataset
    train_set = pd.read_csv(gzip.open(dataset, 'rb'), header=None)

    # Transform attacks into the four main attack classes (Probe, U2R, R2L, DoS)
    train_set = labels_to_classes(train_set)

    # Fit and encode the string features into labels
    train_set = fit_transform(train_set)
    target_class_code = d[41].transform([target_class])[0]
    train_set_filterd = train_set.ix[train_set[41] == target_class_code]

    if classification:
        train_set_outlier = train_set.ix[train_set[41] != target_class_code]
        train_set_outlier.ix[train_set_outlier[41] != target_class_code, 41] = -1
        train_set_filterd.ix[train_set_filterd[41] == target_class_code, 41] = 1
        train_set_filterd = pd.concat([train_set_filterd, train_set_outlier])
    # Return the dataset as a tuple with the following format (features, target)
    train_set_tuple = getTuple(train_set_filterd)

    # Split the training dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(train_set_tuple[0],
                                                      train_set_tuple[1],
                                                      test_size=0.01, stratify=train_set_tuple[1])
    train_set_tuple = X_train, y_train
    val_set_tuple = X_val, y_val

    # Load the test dataset
    test_set = pd.read_csv(gzip.open(test_dataset, 'rb'), header=None)

    # Transform attacks into the four main attack classes (Probe, U2R, R2L, DoS)
    test_set = labels_to_classes(test_set)

    # Remove ICMP records. There are 2 records only in the corrected dataset
    # TODO: Fix this
    test_set = test_set[test_set[2] != 'icmp']

    # Encode the string features into labels
    test_set = transform(test_set)
    test_set_filterd = test_set.ix[test_set[41] == target_class_code]
    if classification:
        test_set_outlier = test_set.ix[test_set[41] != target_class_code]
        test_set_outlier = test_set_outlier.iloc[0:3000]
        test_set_outlier.ix[test_set_outlier[41] != target_class_code, 41] = -1
        test_set_filterd.ix[test_set_filterd[41] == target_class_code, 41] = 1
        test_set_filterd = pd.concat([test_set_filterd, test_set_outlier])

    # Return the dataset as a tuple with the following format (features, target)
    test_set_tuple = getTuple(test_set_filterd)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set_tuple)
    valid_set_x, valid_set_y = shared_dataset(val_set_tuple)
    train_set_x, train_set_y = shared_dataset(train_set_tuple)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
