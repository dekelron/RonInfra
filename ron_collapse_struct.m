% Converts a deep struct into a shallow struct
function out = ron_collapse_struct(in)

    %% Unit test
    
    if nargin==0
        fprintf('Unit testing ron_collapse_struct\n');
        disp("disp(ron_collapse_struct(struct('x', []))):");
        disp(ron_collapse_struct(struct('x', [])));
        disp("disp(ron_collapse_struct(struct('x', [1]))):");
        disp(ron_collapse_struct(struct('x', [1])));
        disp("disp(ron_collapse_struct(struct('x', [1 2 3]))):");
        disp(ron_collapse_struct(struct('x', [1 2 3])));
        disp("disp(ron_collapse_struct(struct('x', {{}}))):");
        disp(ron_collapse_struct(struct('x', {{}})));
        disp("disp(ron_collapse_struct(struct('x', {{1}}))):");
        disp(ron_collapse_struct(struct('x', {{1}})));
        disp("disp(ron_collapse_struct(struct('x', {{1 2 3}}))):");
        disp(ron_collapse_struct(struct('x', {{1 2 3}})));
        disp("disp(ron_collapse_struct(struct('x', {{{}}}))):");
        disp(ron_collapse_struct(struct('x', {{{}}})));
        disp("disp(ron_collapse_struct(struct('x', {{{1}}}))):");
        disp(ron_collapse_struct(struct('x', {{{1}}})));
        disp("disp(ron_collapse_struct(struct('x', {{{1 2 3}}}))):");
        disp(ron_collapse_struct(struct('x', {{{1 2 3}}})));
        disp("disp(ron_collapse_struct(struct('x', {{'bla', 'blu', 'blip'}}))):");
        disp(ron_collapse_struct(struct('x', {{'bla', 'blu', 'blip'}})));
        disp("disp(ron_collapse_struct(struct('x', {{{1 2 3}}}))):");
        disp(ron_collapse_struct(struct('x', {{{{1 2 3}} {1 2} {1}}})));
        return;
    end
    
    
    %% Do work
    
    if ischar(in) || ...
            (isempty(in) && ~isstruct(in)) || ...
            (length(in)==1 && ~iscell(in) && ~isstruct(in))
        out = in;
    elseif isstruct(in) && length(in(:))==1
        % if singleton struct
        out = struct();
        fields = fieldnames(in);
        for field_i = 1:length(fields)
            c_field_name = fields{field_i};
            c_val = in.(c_field_name);
            is_treat_as_arr = length(c_val)>1 || iscell(c_val);
            if isempty(c_val)
                out.(c_field_name) = [];
            elseif ~is_treat_as_arr || ischar(c_val)
                out.(c_field_name) = ron_collapse_struct(c_val);
            elseif length(c_val)>1 || iscell(c_val)
                if ~iscell(c_val), c_val = num2cell(c_val); end
                for arr_i = 1:length(c_val(:)) % Add for cell
                    out.([c_field_name '_' num2str(arr_i)]) = ron_collapse_struct(c_val{arr_i});
                end
            else
                error('Unrecognized case');
            end
        end
    elseif length(in(:))>1 || iscell(in)
        % if length>1 or cell, force to cell array, apply collapse per element
        if ~iscell(in), in = num2cell(in); end % force cell array
        st_vals = cellfun(@ron_collapse_struct, in, 'UniformOutput', false);
        st_field_names = strcat('ind_', arrayfun(@num2str, 1:length(in(:)), 'UniformOutput', false));
        out = cell2struct(st_vals(:), st_field_names(:), 1);
    else
        error('Unrecognized case');
    end
    
end
