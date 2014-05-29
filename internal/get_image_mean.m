function image_mean = get_image_mean(),

persistent image_mean_store;

if isempty(image_mean_store),
  fprintf('cnn: initializing image mean\n');
  input_size = 227;
  mean_image_file = '/data/vision/torralba/hallucination/caffe/ilsvrc_2012_mean.mat';
  image_mean_store = load(mean_image_file);
  image_mean_store = image_mean_store.image_mean;
  off = floor((size(image_mean_store,1) - input_size)/2)+1;
  image_mean_store = image_mean_store(off:off+input_size-1, off:off+input_size-1, :);
end

image_mean = image_mean_store;
