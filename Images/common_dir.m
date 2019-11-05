% finds the longest common dir
function lcd = common_dir(paths)

    if nargin < 1
        fprintf('Unit testing common_dir\n');
        if ispc
            DirsToTest = { ...
                'C:\Program Files (x86)\Windows Media Player', ...
                'C:\Program Files (x86)\Windows Media Player\', ...
                '\\Kislev5\c\Program Files (x86)\Windows Media Player'};
        elseif isunix
            DirsToTest = {'/boot/grub/'};
        end
        for dir_i = 1:length(DirsToTest)
            fprintf('Test %d\n', dir_i);
            c_dir = DirsToTest{dir_i};
            tmp_paths = subdir(c_dir);
            paths = {tmp_paths.name}';
            out_dir = common_dir(paths);
            fprintf('\tInput:  ''%s''\n\tOutput: ''%s''\n', c_dir, out_dir);
        end
        return;
    end
    
    paths = paths(:);
    if isempty(paths)
        longest_idx = 0;
    elseif length(paths(:))==1
        lcd = fileparts(paths{1});
        return;
    else
        if ispc
            broken_paths_tmp = cellfun(@(A) strsplit(A, '\'), paths, 'UniformOutput', false);
        elseif isunix
            broken_paths_tmp = cellfun(@(A) strsplit(A, '/'), paths, 'UniformOutput', false);
        else
            error('Unrecognized architecture');
        end
        lengths = cell2mat(cellfun(@length, broken_paths_tmp, 'UniformOutput', false));
        broken_paths = cell(size(paths, 1), max(lengths));
        for i = 1:size(paths, 1)
            broken_paths(i, 1:lengths(i)) = broken_paths_tmp{i, 1:length(i)};
        end
        num_unique = nan(1, min(lengths));
        for j = 1:min(lengths)
            num_unique(j) = length(unique(broken_paths(:, j)));
        end

        find_last_non1 = find(num_unique>1, 1, 'first');
        if num_unique(1) > 1
            longest_idx = 0;
        elseif all(num_unique==1)
            longest_idx = length(num_unique);
        elseif ~isempty(find_last_non1) && find_last_non1 > 1
            longest_idx = find_last_non1-1;
        else
            error('What does this case mean? Shouldn''t happen.');
        end
    end
    if longest_idx==0
        lcd = '';
    else
        if ispc
            if longest_idx==0
                lcd = '';
            elseif strcmp(paths{1}(1:2), '\\')
                lcd = fullfile('\\', broken_paths{1, 1:longest_idx}, '\'); % net paths
            else
                lcd = fullfile(broken_paths{1, 1:longest_idx}, '\');
            end
        elseif isunix
            if paths{1}(1)=='/'
                lcd = fullfile('/', broken_paths{1, 1:longest_idx}, '/');
            else
                lcd = fullfile(broken_paths{1, 1:longest_idx}, '/');
            end
        else
            error('Unrecognized architecture');
        end
    end
end
