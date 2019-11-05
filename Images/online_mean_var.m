% This class calculates in an online fashion the mean and the var of 
% the input data. It makes the strong assumption of random order in 
% the samples dimension.
% online_mean_var:
%   - sample_size: e.g. [15 30 25] for 15X30X25 sized samples
%   - repeats_dim: e.g. 4, for inputs with repeated samples at fourth dim
%   - max_repeats: e.g. inf
%   - class: e.g. single, double
%   - is_use_gpu_array: false, true
%   - alg: 'original_welford' (default), 'batch_welford'
%
% EXAMPLE for simple use, just do:
%     a = online_mean_var;
%     a.push(samples_batch);
%     a.push(samples_batch2);
%     ...
%     out_mean = a.nanmean;
%     out_var = a.nanvar;
% 
classdef online_mean_var < handle
    properties
        % basic
        is_init
        % info
        sample_size
        repeats_dim
        max_repeats
        class
        is_use_gpu_array
        alg
        % data
        m_n
        m_oldM
        m_newM
        m_oldS
        m_newS
    end
    methods (Static)
        function test()
            %%
            fprintf('Unit testing online_mean_var\n');
            tmp_warn = warning('query', 'online_mean_var:implementation');
            warning('off', 'online_mean_var:implementation');
            N_samples_options = [5, 10, 100, 1000];
            for N_i = 1:length(N_samples_options)
                tmp_obj = online_mean_var();
                c_N = N_samples_options(N_i);
                in = rand(1, c_N);
                for pos_i = 1:size(in, 2)
                    tmp_obj.push(in(pos_i));
                end
                fprintf('%d samples: M=%f, Var=%f, estM=%f, estVar=%f\n', ...
                    c_N, nanmean(in), nanvar(in), tmp_obj.nanmean, tmp_obj.nanvar);
            end
            sz_options = {[1 1 1], [1 500 1], [1 1 500], [500 500 1], ...
                [500 1 500], [1 500 500]};
            flag_options = [0 1];
            gpu_options = [0 1];
            alg_options = {'batch_welford', 'original_welford', 'batch_mean'};
            nan_rates_options = [0 0.01 0.05 0.5];
            for nan_rate_i = 1:length(nan_rates_options)
                c_nan_rate = nan_rates_options(nan_rate_i);
                for gpu_i = 1:length(gpu_options)
                    c_is_use_gpu = gpu_options(gpu_i);
                    for flag_i = 1:length(flag_options)
                        c_flag = flag_options(flag_i);
                        for alg_i = 1:length(alg_options)
                            c_alg = alg_options{alg_i};
                            fprintf('%s, gpu=%d, flag=%d\n', ...
                                c_alg, c_is_use_gpu, c_flag);
                            for sz_i = 1:length(sz_options)
                                c_sz = sz_options{sz_i};
                                tmp_obj = online_mean_var(c_sz, 4, inf, 'double', c_is_use_gpu, c_alg);
                                tmp_obj2 = online_mean_var(1, 4, inf, 'double', c_is_use_gpu, c_alg);
                                tmp_obj3 = online_mean_var(1, 4, inf, 'double', c_is_use_gpu, c_alg);
                                tmp_obj4 = online_mean_var(c_sz, 4, inf, 'double', c_is_use_gpu, c_alg);
                                N_repeats = 5;
                                start_tic = tic;
                                in = rand([c_sz N_repeats]);
                                if c_is_use_gpu
                                    in = gpuArray(in);
                                end
                                if c_nan_rate > 0
                                    in(rand(size(in))<=c_nan_rate) = NaN;
                                end
                                tmp_obj.push(in);
                                tmp_obj2.push(in(1, 1, 1, :));
                                for pos_i = 1:size(in, 4)
                                    tmp_obj3.push(in(1, 1, 1, pos_i));
                                end
                                tmp_obj4.push(in(:, :, :, 1:(floor(N_repeats/2)-1)));
                                tmp_obj4.push(in(:, :, :, floor(N_repeats/2):end));
                                switch c_alg
                                    case {'original_welford', 'batch_welford'}
                                        a = tmp_obj.nanvar;
                                        b = tmp_obj4.nanvar;
                                        if abs(a(1) - tmp_obj2.nanvar) > 0.01
                                            error('Dim expansion error');
                                        end
                                        if abs(a(1) - tmp_obj3.nanvar) > 0.01
                                            error('Dim expansion error');
                                        end
                                        if abs(a(1) - b(1)) > 0.01
                                            error('Dim expansion error');
                                        end
                                    case 'batch_mean'
                                    otherwise
                                        error('Unrecognized option');
                                end
                                tmp_esp = 0.00001;
                                max_abs_mean_err = max(max(max(max(abs(nanmean(in, 4) - tmp_obj.nanmean)))));
                                assert(isequaln(nanmean(in, 4), tmp_obj.nanmean) || ...
                                    max_abs_mean_err < tmp_esp);
                                switch c_alg
                                    case {'original_welford', 'batch_welford'}
                                        max_abs_var_err = max(max(max(max(abs(nanvar(in, c_flag, 4) - tmp_obj.nanvar(c_flag))))));
                                        assert(isequaln(nanvar(in, c_flag, 4), tmp_obj.nanvar(c_flag)) || ...
                                            max_abs_var_err < tmp_esp);
                                    case 'batch_mean'
                                        max_abs_var_err = NaN;
                                    otherwise
                                        error('Unrecognized option');
                                end
                                fprintf(...
                                    ['\t\t\tOK, sz of samples (%d %d %d)X%d    : max abs err for mean: %f, ' ...
                                    'for variance: %f, took %.2f seconds\n'], ...
                                    c_sz(1), c_sz(2), c_sz(3), N_repeats, max_abs_mean_err, ...
                                    max_abs_var_err, toc(start_tic));
                                max_abs_mean_err = max(max(max(max(abs(nanmean(in, 4) - tmp_obj4.nanmean)))));
                                assert(isequaln(nanmean(in, 4), tmp_obj4.nanmean) || ...
                                    max_abs_mean_err < tmp_esp);
                                switch c_alg
                                    case {'original_welford', 'batch_welford'}
                                        max_abs_var_err = max(max(max(max(abs(nanvar(in, c_flag, 4) - tmp_obj4.nanvar(c_flag))))));
                                        assert(isequaln(nanvar(in, c_flag, 4), tmp_obj4.nanvar(c_flag)) || ...
                                            max_abs_var_err < tmp_esp);
                                    case 'batch_mean'
                                        max_abs_var_err = NaN;
                                    otherwise
                                        error('Unrecognized option');
                                end
                                fprintf(...
                                    ['\t\t\tOK, sz of samples (%d %d %d)X%d-TWO: max abs err for mean: %f, ' ...
                                    'for variance: %f, took %.2f seconds\n'], ...
                                    c_sz(1), c_sz(2), c_sz(3), N_repeats, max_abs_mean_err, ...
                                    max_abs_var_err, toc(start_tic));
                            end
                        end
                    end
                end
            end
            warning(tmp_warn.state, 'online_mean_var:implementation');
            fprintf('All tests OK\n');
        end
    end
    methods
        function obj = online_mean_var( ...
                sample_size, repeats_dim, max_repeats, ...
                class, is_use_gpu_array, alg)
            if nargin == 0 && nargout == 0
                online_mean_var.test();
            elseif nargin == 0
                obj.is_init = false;
            else
                if nargin < 2, repeats_dim = length(sample_size)+1; end
                if nargin < 3, max_repeats = inf; end
                if nargin < 4, class = 'double'; end
                if nargin < 5, is_use_gpu_array = false; end
                if nargin < 6, alg = 'original_welford'; end
                obj.is_init = false;
                obj = init(obj, sample_size, repeats_dim, ...
                    max_repeats, class, is_use_gpu_array, alg);
                obj.is_init = true;
            end
        end
        function obj = init(obj, sample_size, repeats_dim, ...
                max_repeats, class, is_use_gpu_array, alg)
            % defaults
            assert(obj.is_init == false);
            assert(nargin >= 2);
            if nargin < 3, repeats_dim = length(sample_size)+1; end
            if nargin < 4, max_repeats = inf; end
            if nargin < 5, class = 'double'; end
            if nargin < 6, is_use_gpu_array = false; end
            if nargin < 7, alg = 'original_welford'; end
            % info
            assert(repeats_dim > length(sample_size) || ...
                sample_size(repeats_dim)==1);
            obj.max_repeats = max_repeats;
            obj.sample_size = sample_size;
            obj.repeats_dim = repeats_dim;
            obj.class = class;
            obj.is_use_gpu_array = is_use_gpu_array;
            obj.alg = alg;
            % data
            obj.m_n = zeros(sample_size, 'uint32');
            switch alg
                case {'original_welford', 'batch_welford'}
                    if is_use_gpu_array
                        obj.m_oldM = gpuArray(nan(sample_size, obj.class));
                        obj.m_newM = gpuArray(nan(sample_size, obj.class));
                        obj.m_oldS = gpuArray(nan(sample_size, obj.class));
                        obj.m_newS = gpuArray(nan(sample_size, obj.class));
                    else
                        obj.m_oldM = nan(sample_size, obj.class);
                        obj.m_newM = nan(sample_size, obj.class);
                        obj.m_oldS = nan(sample_size, obj.class);
                        obj.m_newS = nan(sample_size, obj.class);
                    end
                case 'batch_mean'
                    if is_use_gpu_array
                        obj.m_newM = gpuArray(nan(sample_size, obj.class));
                    else
                        obj.m_newM = nan(sample_size, obj.class);
                    end
                otherwise
                    error('Unrecognized alg');
            end
            warning('online_mean_var:implementation', ...
                'This class assumes that input samples are randomly placed');
        end
        function obj = push(obj, samples)
            % Can make more accurate and efficient by mean and var across repeats_dim
            % then combining with old
            if obj.is_init == false
                tmp_sample_size = size(samples);
                obj = init(obj, tmp_sample_size);
                obj.is_init = true;
            end
            switch obj.alg
                case 'original_welford'
                    % Based on http://www.johndcook.com/blog/standard_deviation/
                    for repeat_i = 1:size(samples, obj.repeats_dim)
                        tmp_subs = repmat({':'}, 1, length(size(samples)));
                        if size(samples, obj.repeats_dim) > 1
                            tmp_subs{obj.repeats_dim} = repeat_i;
                        end
                        S = struct();
                        S.type = '()';
                        S.subs = tmp_subs;
                        c_sample = cast(subsref(samples, S), obj.class);
                        for dim_i = 1:length(obj.sample_size)
                            assert(size(c_sample, dim_i)==obj.sample_size(dim_i));
                        end
                        assert(size(c_sample, obj.repeats_dim)==1);
                        ok = gather(~isnan(c_sample));
                        ok1 = gather(ok & obj.m_n==0);
                        ok2p = gather(ok & obj.m_n>0);
                        if any(ok) % update not-nan
                            obj.m_n(ok) = obj.m_n(ok) + 1;
                        end
                        if any(ok1) % update not-nan and first
                            obj.m_oldM(ok1) = c_sample(ok1);
                            obj.m_newM(ok1) = c_sample(ok1);
                            obj.m_oldS(ok1) = 0;
                            obj.m_newS(ok1) = 0;
                        end
                        if any(ok2p) % update not-nan and not-first
                            obj.m_newM(ok2p) = obj.m_oldM(ok2p) + ...
                                (c_sample(ok2p) - obj.m_oldM(ok2p)) ./ ...
                                cast(obj.m_n(ok2p), obj.class);
                            obj.m_newS(ok2p) = obj.m_oldS(ok2p) + ...
                                (c_sample(ok2p) - obj.m_oldM(ok2p)) .* ...
                                (c_sample(ok2p) - obj.m_newM(ok2p));
                        end
                        % set up for next iteration
                        obj.m_oldM = obj.m_newM;
                        obj.m_oldS = obj.m_newS;
                    end
                    
                case 'batch_welford'
                    samples = cast(samples, obj.class);
                    samples_m_n = cast(sum(~isnan(samples), obj.repeats_dim), 'uint32');
                    samples_mean = nanmean(samples, obj.repeats_dim);
                    ok = gather(samples_m_n > 0);
                    ok1 = gather(ok & obj.m_n == 0);
                    ok2p = gather(ok & obj.m_n > 0);
                    obj.m_n = obj.m_n + samples_m_n;
                    if any(ok1) % update not-nan and first
                        obj.m_newM(ok1) = samples_mean(ok1);
                        obj.m_oldM(ok1) = samples_mean(ok1);
                    end
                    if any(ok2p) % update not-nan and not-first
                        obj.m_newM(ok2p) = obj.m_oldM(ok2p) + ...
                            (samples_mean(ok2p) - obj.m_oldM(ok2p)) .* ...
                                (cast(samples_m_n(ok2p), obj.class) ./ ...
                                cast(obj.m_n(ok2p), obj.class));
                    end
                    samples_S = nansum( ...
                        bsxfun(@minus, samples, obj.m_oldM) .* ...
                        bsxfun(@minus, samples, obj.m_newM), ...
                        obj.repeats_dim);
                    if any(ok1)
                        obj.m_newS(ok1) = samples_S(ok1);
                    end
                    if any(ok2p) 
                        obj.m_newS(ok2p) = obj.m_oldS(ok2p) + samples_S(ok2p);
                    end
                    % set up for next iteration
                    obj.m_oldM = obj.m_newM;
                    obj.m_oldS = obj.m_newS;
                    
                case 'batch_mean'
                    samples = cast(samples, obj.class);
                    samples_m_n = cast(sum(~isnan(samples), obj.repeats_dim), 'uint32');
                    samples_mean = nanmean(samples, obj.repeats_dim);
                    ok = gather(samples_m_n > 0);
                    ok1 = gather(ok & obj.m_n == 0);
                    ok2p = gather(ok & obj.m_n > 0);
                    obj.m_n = obj.m_n + samples_m_n;
                    if any(ok1) % update not-nan and first
                        obj.m_newM(ok1) = samples_mean(ok1);
                    end
                    if any(ok2p) % update not-nan and not-first
                        obj.m_newM(ok2p) = obj.m_newM(ok2p) + ...
                            (samples_mean(ok2p) - obj.m_newM(ok2p)) .* ...
                                (cast(samples_m_n(ok2p), obj.class) ./ ...
                                cast(obj.m_n(ok2p), obj.class));
                    end
                    
                otherwise
                    error('Unrecognized option');
            end
        end
        function m_n = num_values(obj)
            m_n = obj.m_n;
        end
        function out_mean = nanmean(obj)
            out_mean = obj.m_newM;
            out_mean(obj.m_n == 0) = NaN;
        end
        function out_var = nanvar(obj, flag)
            switch obj.alg
                case {'original_welford', 'batch_welford'} % ok
                case 'batch_mean', error('To get var use welford alg');
                otherwise, error('Unrecognized option');
            end
            if nargin==1, flag=0; end
            if flag==0
                out_var = obj.m_newS ./ cast(obj.m_n - 1, obj.class);
            elseif flag==1
                out_var = obj.m_newS ./ cast(obj.m_n, obj.class);
            else
                error('Unrecognized flag');
            end
            out_var(obj.m_n <= 1) = 0; % To be like MATLAB functions
        end
        function out_std = nanstd(obj, flag)
            if nargin==1, flag=0; end
            out_std = sqrt(obj.nanvar(flag));
        end
    end
end

