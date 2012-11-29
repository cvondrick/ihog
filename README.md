iHOG: Inverting Histograms of Oriented Gradients
================================================

This software package contains tools to invert and visualize HOG features.
It implements the Paired Dictionary Learning algorithm described in our
paper "Inverting and Visualizing Features for Object Detection" [1].

Inverting HOG
-------------

To invert a HOG point, use the 'invertHOG()' function:

    >> feat = features(im, 8);
    >> ihog = invertHOG(feat);
    >> imagesc(ihog); axis image;

Our library will automatically load the necessary data files to perform
the inversion and caches them into memory to reduce IO.


Visualizing HOG
---------------

In addition to 'invertHOG()', we provide a 'visualizeHOG()' function that
automatically shows a figure with a few visualizations of the HOG feature point:

    >> feat = features(im, 8);
    >> visualizeHOG(feat);

'visualizeHOG()' can also show you more visualizations. Specify an optional second
parameter to increase the verbosity level:

    >> visualizeHOG(feat, 1);
    >> visualizeHOG(feat, 2);
    >> visualizeHOG(feat, 3);


Installation
------------

Before you can use this tool, you must compile iHOG. Execute the 'compile'
script in MATLAB:

    >> compile

This command will compile the HOG feature extraction code and the sparse coding
SPAMS toolbox.

When you use iHOG, remember to adjust your MATLAB path:

    >> addpath(genpath('/path/to/ihog'))

Otherwise, iHOG will be unable to find relevant libraries.

Learning
--------

We provide a prelearned dictionary in 'pd.mat', but you can learn your own if
you wish. Simply call the 'learnpairdict()' function and pass it a directory of
images:

    >> pd = learnpairdict('/path/to/images/', 1000000, 1000, 5, 5);

The above learns a 5x5 HOG patch paired dictionary with 1000 elements and a training
set size of one million window patches. Depending on the size of the problem, it may
take minutes or hours to complete.

References
----------

[1] Carl Vondrick, Aditya Khosla, Tomasz Malisiewicz, Antonio Torralba.
"Inverting and Visualizing Features for Object Detection." Technical Report.
2013.
