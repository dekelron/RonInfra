function out_img = ron_format_conversion(in_img, output_format, input_format, max_alpha)

    if nargin==0
        %% Test
        tmp_img_gray = imread('cameraman.tif');
        tmp_img_color = imread('onion.png');
        disp(size(ron_format_conversion(tmp_img_gray, 'force_3channels')));
        disp(size(ron_format_conversion(tmp_img_color, 'force_3channels')));
        return;
    end

    %% Parse inputs
    
    % output_format
    if nargin==1
        error('Please specify output_format argument');
    end
    
    % input_format
    if nargin==2 || isempty(input_format)
        switch size(in_img, 3)
            case 1, input_format = 'gray';
            case 3, input_format = 'RGB';
            case 4
                switch output_format
                    case 'gray',                    input_format = 'RGBA_straightAlpha'; % default is RGBA_straightAlpha
                    case 'RGB',                     input_format = 'RGBA_straightAlpha'; % default is RGBA_straightAlpha
                    case 'RGBA_straightAlpha',      input_format = 'RGBA_straightAlpha';
                    case 'RGBA_premultipliedAlpha', input_format = 'RGBA_premultipliedAlpha';
                    otherwise, error('Unrecognized output_format=%s', output_format);
                end
            otherwise
                error('Unrecognized input type, cannot guess input_format');
        end
    end
    switch input_format
        case 'gray',                    assumed_colordepth = 1;
        case 'RGB',                     assumed_colordepth = 3;
        case 'RGBA_straightAlpha',      assumed_colordepth = 4;
        case 'RGBA_premultipliedAlpha', assumed_colordepth = 4;
        otherwise, error('Unrecognized input_format=%s', input_format);
    end
    if size(in_img, 3) ~= assumed_colordepth
        error('For input_format=%s we assume colordepth=%d but in_img has colordepth=%d', ...
            input_format, assumed_colordepth, size(in_img, 3));
    end
    
    
    %% Convert
    
    switch output_format
        case 'gray'
            switch input_format
                case 'gray'
                    out_img = in_img;
                case {'RGB', 'RGBA_premultipliedAlpha'}
                    out_img = in_img(:, :, 1, :);
                    out_img(:) = NaN;
                    for img_i = 1:size(in_img, 4)
                        out_img(:, :, 1, img_i) = rgb2gray(in_img(:, :, 1:3, img_i));
                    end
                case 'RGBA_straightAlpha'
                    out_img = in_img(:, :, 1, :);
                    out_img(:) = NaN;
                    for img_i = 1:size(in_img, 4)
                        out_img(:, :, 1, img_i) = double(rgb2gray(in_img(:, :, 1:3, img_i))) .* (double(in_img(:, :, 4, img_i))/max_alpha);
                    end
                otherwise
                    error('Unrecognized input_format=%s', input_format);
            end
        case 'RGB'
            switch input_format
                case 'gray'
                    out_img = repmat(in_img, [1 1 3 1]);
                case 'RGB'
                    out_img = in_img;
                case 'RGBA_straightAlpha'
                    out_img = in_img(:, :, 1:3, :); % To keep the format
                    out_img(:, :, 1:3, :) = bsxfun(@times, double(in_img(:, :, 1:3, :)), double(in_img(:, :, 4, :))/max_alpha);
                case 'RGBA_premultipliedAlpha'
                    out_img = in_img(:, :, 1:3, :);
                otherwise
                    error('Unrecognized input_format=%s', input_format);
            end
        case 'RGBA_straightAlpha'
            switch input_format
                case 'gray'
                    out_img = repmat(in_img, [1 1 4 1]);
                    out_img(:, :, 4, :) = max_alpha;
                case 'RGB'
                    out_img = in_img(:, :, 1:3, :);
                    out_img(:, :, 4, :) = max_alpha;
                case 'RGBA_straightAlpha'
                    out_img = in_img;
                case 'RGBA_premultipliedAlpha'
                    out_img = in_img;
                    out_img(:, :, 1:3, :) = bsxfun(@divide, double(out_img(:, :, 1:3, :)), double(out_img(:, :, 4, :))/max_alpha);
                otherwise
                    error('Unrecognized input_format=%s', input_format);
            end
        case 'RGBA_premultipliedAlpha'
            switch input_format
                case 'gray'
                    out_img = repmat(in_img, [1 1 4 1]);
                    out_img(:, :, 4, :) = max_alpha;
                case 'RGB'
                    out_img = in_img(:, :, 1:3, :);
                    out_img(:, :, 4, :) = max_alpha;
                case 'RGBA_straightAlpha'
                    out_img = in_img;
                    out_img(:, :, 1:3, :) = bsxfun(@times, double(out_img(:, :, 1:3, :)), double(out_img(:, :, 4, :))/max_alpha);
                case 'RGBA_premultipliedAlpha'
                    out_img = in_img;
                otherwise
                    error('Unrecognized input_format=%s', input_format);
            end
        otherwise
            error('Unrecognized output_format=%s', output_format);
    end

end
