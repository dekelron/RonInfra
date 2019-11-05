% Reqursively merge many kinds of variable types
% Overwrites on a
function out = ron_deep_merge(a, b)

    %% Unit testing
    
    if nargin==0
        disp('ron_deep_merge({1 2 3},{1 3}''):');
         disp(ron_deep_merge({1 2 3},{1 3}'));
        disp('ron_deep_merge({1 2 []},{1 [] 3}):');
         disp(ron_deep_merge({1 2 []},{1 [] 3}));
        disp('ron_deep_merge(struct(''f1'', 1), struct(''f2'', 5)):');
        tmp = ron_deep_merge(struct('f1', {1 2}), struct('f2', 5));
        disp(tmp);
        disp(tmp(1));
        disp(tmp(2));
        return;
    end

    
    %% Merge
    
    if isequal(a,b)
        out = b;
    elseif isempty(a)
        out = b;
    elseif isempty(b)
        out = a;
    elseif iscell(a)
        if ~iscell(b)
            error('for deep merge, if a is a cell array, b must be cell array as well');
        end
        % Address difference in sizes
        out_sz = nan(1, max(length(size(a)),length(size(b))));
        for dim_i = 1:length(out_sz)
            out_sz(dim_i) = max(size(a, dim_i), size(b, dim_i));
        end
        if length(out_sz)>5, error('Not implemented'); end
        out_sz((length(out_sz)+1):5) = 1; % Size of 1 where undefined
        if any(size(a) ~= out_sz(1:length(size(a)))) % Extend a
            a{out_sz(1), out_sz(2), out_sz(3), out_sz(4), out_sz(5)} = [];
        end
        if any(size(b) ~= out_sz(1:length(size(b)))) % Extend a
            b{out_sz(1), out_sz(2), out_sz(3), out_sz(4), out_sz(5)} = [];
        end
        % merge
        out = cellfun(@ron_deep_merge, a, b, 'UniformOutput', false);
    elseif isstruct(a)
        assert(isstruct(b), 'for deep merge, if a is a struct, b must be struct as well');
        % Address difference in field names
        a_fields = fieldnames(a);
        b_fields = fieldnames(b);
        b_fields_not_in_a = setdiff(b_fields, a_fields);
        for field_i = 1:length(b_fields_not_in_a)
            a(1).(b_fields_not_in_a{field_i}) = [];
        end
        a_fields_not_in_b = setdiff(a_fields, b_fields);
        for field_i = 1:length(a_fields_not_in_b)
            b(1).(a_fields_not_in_b{field_i}) = [];
        end
        b = orderfields(b, a); % order fields in b like in a
        assert(isequal(fieldnames(a), fieldnames(b)));
        % Address difference in sizes
        empty_st = cell2struct(cell(size(fieldnames(a))), fieldnames(a), 1);
        out_sz = nan(1, max(length(size(a)),length(size(b))));
        for dim_i = 1:length(out_sz)
            out_sz(dim_i) = max(size(a, dim_i), size(b, dim_i));
        end
        if length(out_sz)>5, error('Not implemented'); end
        out_sz((length(out_sz)+1):5) = 1; % Size of 1 where undefined
        if any(size(a) ~= out_sz(1:length(size(a)))) % Extend a
            a(out_sz(1), out_sz(2), out_sz(3), out_sz(4), out_sz(5)) = empty_st;
        end
        if any(size(b) ~= out_sz(1:length(size(b)))) % Extend b
            b(out_sz(1), out_sz(2), out_sz(3), out_sz(4), out_sz(5)) = empty_st;
        end
        % merge
        out = cell2struct(ron_deep_merge(struct2cell(a), struct2cell(b)), fieldnames(a), 1);
    else
        disp('disp(a):'); disp(a);
        disp('disp(b):'); disp(b);
        error('the difference between a to b is irreconcilable');
    end


end