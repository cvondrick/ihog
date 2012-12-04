iHOG: Inverting Histograms of Oriented Gradients
================================================

This software package contains tools to invert and visualize HOG features.
It implements the Paired Dictionary Learning algorithm described in our
paper "Inverting and Visualizing Features for Object Detection" [1].

Installation
------------

Before you can use this tool, you must compile iHOG. Execute the 'compile'
script in MATLAB to compile the HOG feature extraction code and sparse coding
SPAMS toolbox:

    $ cd /path/to/ihog
    $ matlab
    >> compile
    
Remember to also adjust your path so MATLAB can find iHOG:

    >> addpath(genpath('/path/to/ihog'))

If you want to use iHOG in your own project, you can simply drop the iHOG
directory into the root of your project.

Inverting HOG
-------------

To invert a HOG point, use the 'invertHOG()' function:

    >> feat = features(im, 8);
    >> ihog = invertHOG(feat);
    >> imagesc(ihog); axis image;

Computing the inverse should take no longer than a second for a typical sized
image on a modern computer.

Visualizing HOG
---------------

iHOG has functions to visualize HOG. The most basic is 'visualizeHOG()':

    >> feat = features(im, 8);
    >> visualizeHOG(feat);

The above displays a figure with the HOG glyph and the HOG inverse. This
visualization is a drop-in replacement for more standard visualizations, and
should work with existing code bases.

Other visualizations are available. Check out the 'visualizations/' folder and
read the comments for more.

Learning
--------

We provide a prelearned dictionary in 'pd.mat', but you can learn your own if
you wish. Simply call the 'learnpairdict()' function and pass it a directory of
images:

    >> pd = learnpairdict('/path/to/images/', 1000000, 1000, 5, 5);

The above learns a 5x5 HOG patch paired dictionary with 1000 elements and a
training set size of one million window patches. Depending on the size of the
problem, it may take minutes or hours to complete.

Bundled Libraries
-----------------

The iHOG package contains source code from the SPAMS sparse coding toolbox
(http://spams-devel.gforge.inria.fr/). We have modified their code to better
support 64 bit machines.

In addition, we have included a select few files from the discriminatively
trained deformable parts model (http://people.cs.uchicago.edu/~rbg/latent/).
We use their HOG computation code and glyph visualization code.

Questions and Comments
----------------------

If you have any feedback, please write to Carl Vondrick at <vondrick@mit.edu>.

References
----------

The conference paper for this software is currently under submission. In
the mean time, please see our technical report:

[1] Carl Vondrick, Aditya Khosla, Tomasz Malisiewicz, Antonio Torralba.
"Inverting and Visualizing Features for Object Detection." Technical Report.
2013.
