% Resize as in AlexNet (which maintains aspect ratio)
function out_img = resize_image(in_img, TargetSize, ResizeConfig)

    if nargin < 3, ResizeConfig = {}; end

    if length(TargetSize)>2
        error('Not implemented, use length(TargetSize)= 1 or 2');
    end
    if length(TargetSize)==1
        TargetSize = [TargetSize TargetSize];
    end

    size_1 = size(in_img, 1);
    size_2 = size(in_img, 2);
    ratio = max(TargetSize ./ [size_1 size_2]);
    tmp_img = imresize(in_img, ratio, ResizeConfig{:});
    pad_in_px = ceil(([size(tmp_img,1) size(tmp_img,2)] - TargetSize) / 2);
    rect = [1+pad_in_px(2) 1+pad_in_px(1) TargetSize-1 TargetSize-1]; % [xmin ymin width height]
    if 0
        out_img = imcrop(tmp_img, rect);
    else
        dim1_range = rect(2):rect(2)+rect(4);
        dim2_range = rect(1):rect(1)+rect(3);
        out_img = tmp_img(dim1_range, dim2_range, :, :);
    end
    if size(out_img, 1) ~= TargetSize(1) || ...
            size(out_img, 2) ~= TargetSize(2) || ...
            size(out_img, 3) ~= size(in_img, 3) || ...
            size(out_img, 4) ~= size(in_img, 4)
        error('Resizing error');
    end
    if 0
        figure(1); clf;
        subplot(1, 2, 1); imshow(in_img);
        subplot(1, 2, 2); imshow(out_img);
    end
end