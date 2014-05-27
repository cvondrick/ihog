chunkmasterfile = '/data/vision/torralba/hallucination/icnn/rcnn-features-chunks-pad/master.mat';
k = 2048;

pd = learnCNNdict(chunkmasterfile, k, .02, 20),

save('pd-caffe-pad.mat', '-struct', 'pd');
