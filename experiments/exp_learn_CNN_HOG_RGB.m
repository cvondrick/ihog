chunkmasterfile = '/data/vision/torralba/hallucination/icnn/rcnn-features-chunks-hog/master.mat';
k = 2048;

pd = learnCNNdict(chunkmasterfile, k, .02, 0),

save('pd-cnn-hog-rgb.mat', '-struct', 'pd');
