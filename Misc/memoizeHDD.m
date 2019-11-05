% memoizeHDD: resembles MATLAB "memoize" but uses files on HDD rather than RAM.
% There are some limitations (see below), so use intelligently.
% Uses DataHash function, please obtain separately: https://www.mathworks.com/matlabcentral/fileexchange/31272-datahash
%
% Example 1:
%   slow_func = @(n) sum(reshape(pca(magic(n)), [], 1))
%   slow_func_memoized = memoizeHDD(slow_func);
%   tic; slow_func_memoized(1000), toc; % First call:  MISS
%   tic; slow_func_memoized(1000), toc; % Second call: HIT
%   tic; slow_func_memoized(1001), toc; % New call:    MISS
% Example 2:
%   corr_memoized = memoizeHDD(@corr);
%   r = corr_memoized((1:100)', (1:100).^2'),                         % MISS
%   [r,p] = corr_memoized((1:100)', (1:100).^2'),                     % MISS
%   [r,p] = corr_memoized((1:100)', (1:100).^2', 'type', 'Pearson'),  % MISS
%   [r,p] = corr_memoized((1:100)', (1:100).^2', 'type', 'Spearman'), % MISS
%   r = corr_memoized((1:100)', (1:100).^2'),                         % HIT
%   [r,p] = corr_memoized((1:100)', (1:100).^2'),                     % HIT
%   [r,p] = corr_memoized((1:100)', (1:100).^2', 'type', 'Pearson'),  % HIT
%   [r,p] = corr_memoized((1:100)', (1:100).^2', 'type', 'Spearman'), % HIT
% Example 3:
%   rand_mem = memoizeHDD(@(a) rand(a))
%   disp(rand_mem(2));
%   disp(rand_mem(3));
%   disp(rand_mem(2)); % Oh no!
%
% Limitations:
%    - Uses MATLAB "save" and "load" which limits what can be saved.
%    - Does not delete unused cache entries, even if HDD is full.
% Features:
%    - Tries to track m-file changes. When a function is changed, the cache should be automatically renewed.
%    - Tries to randomly verify that there is no catastrophe. (see below, can disable by setting use_cache_prob to 1)
%
% Date: 18 Nov 2017, update 05 Nov 2019
% License: MIT
%
function varargout = memoizeHDD( ...
        func, ...              % MANDATORY - what function to cache
        cache_root_dir, ...    % OPTIONAL  - directory path for cache
        use_cache_prob, ...    % OPTIONAL - probability for use of cache, values: [0,1], dafult: 0.99, useful as a random verification
        is_function_call, ...  % DO NOT USE
        func_hash, ...         % DO NOT USE
        varargin ...           % DO NOT USE
        )

    % func
    assert(nargin>0, 'Must provide at least one input argument');
    assert(isa(func, 'function_handle'), 'First input argument must be a function handle');
    
    % cache_root_dir
    if nargin<2 || isempty(cache_root_dir)
        if ispc
            cache_root_dir = 'C:\tmp\memoizeHDD\';
            [~, ~, ~] = mkdir('C:\tmp');
            [~, ~, ~] = mkdir('C:\tmp\memoizeHDD\');
        elseif isunix || ismac
            cache_root_dir = '/tmp/memoizeHDD/';
            [~, ~, ~] = mkdir('/tmp/');
            [~, ~, ~] = mkdir('/tmp/memoizeHDD/');
        else
            error('Unrecognized architecture');
        end
    end
    
    % verification_prob
    if nargin < 3
        use_cache_prob = 0.99;
    end
    
    % cache_dir
    subdir_name_tmp = matlab.lang.makeValidName(func2str(func));
    subdir_name = subdir_name_tmp(1:min(end,25)); % First 25 characters
    cache_dir = fullfile(cache_root_dir, subdir_name);
    [~, ~, ~] = mkdir(cache_dir);
    if ~(exist(cache_dir, 'dir')==7)
        error('Failed to find or create cache directory %s', cache_dir);
    end
    
    % Do work
    is_debug = false;
    is_debug = is_debug; %#ok
    if nargin<4, is_function_call = false; end
    if is_function_call
        % Function call
        cache_id = struct();
        cache_id.varargin = varargin;
        cache_id.nargout = nargout;
        cache_id.func_hash = func_hash;
        cache_md5 = DataHash(cache_id);
        cache_md5_start = cache_md5(1:3);
        cache_md5_end = cache_md5(4:end);
        possible_cache_subdir = fullfile(cache_dir, cache_md5_start);
        possible_cache_path = fullfile(possible_cache_subdir, [cache_md5_end '.mat']);
        if use_cache_prob>0 && exist(possible_cache_subdir, 'dir')==7 && exist(possible_cache_path, 'file')==2
            % MD5 HIT
            try
                loaded_data = load(possible_cache_path);
            catch
                loaded_data = struct();
            end
            if isfield(loaded_data, 'cache_id') && isequal(loaded_data.cache_id, cache_id) && ...
                    isfield(loaded_data, 'varargout') && iscell(loaded_data.varargout)
                % Exact HIT
                varargout = loaded_data.varargout;
                if is_debug, fprintf('Cache HIT\n'); end
                if use_cache_prob<1 && rand>use_cache_prob
                    % Verify sometimes. Hopefully a good practice at a negligible cost.
                    tmp_varargout = varargout;
                    [varargout{1:nargout}] = func(varargin{:});
                    if ~isequaln(tmp_varargout, varargout)
                        error('DAMN!! Caching does NOT work, as indicated by a random verification');
                    end
                end
            else
                % MISS, despite the MD5 hit
                [varargout{1:nargout}] = func(varargin{:});
                save(possible_cache_path, 'cache_id', 'varargout');
                if is_debug, fprintf('Cache MISS, despite MD5 hit\n'); end
            end
        else
            % MISS
            [~, ~, ~] = mkdir(possible_cache_subdir);
            [varargout{1:nargout}] = func(varargin{:});
            save(possible_cache_path, 'cache_id', 'varargout');
            if is_debug, fprintf('Cache MISS\n'); end
        end
    else
        % Try to identify potential dependencies (only once per MATLAB run)
        try
            get_dep_func = @matlab.codetools.requiredFilesAndProducts;
            try
                get_dep_func_memoize = memoize(get_dep_func);
            catch
                get_dep_func_memoize = get_dep_func;
            end
            dependencies_paths = get_dep_func_memoize(func2str(func));
            dependencies_data = cell(size(dependencies_paths));
            for path_i = 1:length(dependencies_paths)
                try
                    dependencies_data{path_i} = fileread(dependencies_paths{path_i});
                catch
                end
            end
            func_hash = DataHash(dependencies_data);
        catch
            func_hash = 0;
        end
        % Handle definition
        varargout = {@(varargin) memoizeHDD( ...
            func, cache_root_dir, use_cache_prob, 1, func_hash, varargin{:})};
        fprintf('Caching results for function "%s" in directory "%s" using func_hash="%s"\n', ...
            func2str(func), cache_dir, func_hash);
    end
end
