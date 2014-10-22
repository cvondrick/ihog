chunkmasterfile = '/data/vision/torralba/hallucination/icnn/rcnn-features-chunks-lab/master.mat';
k = 1024;

pd = learnCNNdict(chunkmasterfile, k, .01, 10),

save('pd-caffe-lab.mat', '-struct', 'pd');
