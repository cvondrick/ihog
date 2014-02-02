iHOG: Inverting Histograms of Oriented Gradients
================================================

This software package contains tools to invert and visualize HOG features.
It implements the Paired Dictionary Learning algorithm described in our
paper "HOGgles: Visualizing Object Detection Features" [1].

Installation
------------

Before you can use this tool, you must compile iHOG. Execute the 'compile'
script in MATLAB to compile the HOG feature extraction code and sparse coding
SPAMS toolbox:

    $ cd /path/to/ihog
    $ matlab
    >> compile
    
If you run into trouble compiling the SPAMS code, you might try opening 
the file `/path/to/ihog/spams/compile.m` and adjusting the settings for
your computer.
    
Remember to also adjust your path so MATLAB can find iHOG:

    >> addpath(genpath('/path/to/ihog'))

If you want to use iHOG in your own project, you can simply drop the iHOG
directory into the root of your project.

In order to use iHOG, you must have a learned paired dictionary. By default,
iHOG will attempt to download a pretrained one from MIT for you on the first
execution. If you wish to download it manually, simply do:

    $ wget http://people.csail.mit.edu/vondrick/pd.mat

Inverting HOG
-------------

To invert a HOG point, use the 'invertHOG()' function:

    >> feat = features(im, 8);
    >> ihog = invertHOG(feat);
    >> imagesc(ihog); axis image;

Computing the inverse should take no longer than a second for a typical sized
image on a modern computer. (It may slower the first time you invoke it as it
caches the paired dictionary from disk.)

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

If you use our software, please cite our conference paper:

[1] Carl Vondrick, Aditya Khosla, Tomasz Malisiewicz, Antonio Torralba.
"HOGgles: Visualizing Object Detection Features"  International Conference
on Computer Vision (ICCV), Sydney, Australia, December 2013.
