% Get a cell array of a subfield.
% For 1-d structs, can just do: {st.a}
% For n-d structs, use this function! (But up-to ~15 dims becuase of my lazyness.)
function out_cell = ron_getfield_arr(struct_arr, field_name)

    %% Unit test
    
    if nargin==0
        fprintf('Unit testing ron_getfield_arr\n');
        tmp_struct_arr = repmat(struct('a', 5, 'b', 7), [3 3]);
        tmp_out = ron_getfield_arr(tmp_struct_arr, 'a');
        disp(tmp_out);
        return;
    end

    %% Go
    
    % Parse
    if ~isstruct(struct_arr)
        error('input struct array must be a struct');
    end
    if ~ischar(field_name)
        error('Field name must be a character array');
    end

    % Extract
    sub_field_names = strsplit(field_name, '.');
    if length(sub_field_names)==1
        % Old code
        tmp_cell = struct2cell(struct_arr);
        names = fieldnames(struct_arr);
        idx = find(strcmp(names, field_name));
        if isempty(idx)
            error('Required field "%s" is not a field of input struct', field_name);
        end
        if length(idx) > 1
            error('Unknown bug.');
        end
		if numel(size(tmp_cell))>15, error('Not implemented?'); end
        out_cell(:, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :) = shiftdim( ...
            tmp_cell(idx, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :), 1);
    else
        % Extended code
        % Go over subfields separated by ".", for example st.field1.field2.field3
        tmp_state = struct_arr;
        for sub_i = 1:length(sub_field_names)
            assert(iscell(tmp_state)||isstruct(tmp_state));
            if iscell(tmp_state)
                for cell_i = 1:length(tmp_state(:))
                    tmp_state{cell_i} = tmp_state{cell_i}.(sub_field_names{sub_i});
                end
            elseif isstruct(tmp_state)
                tmp_state = ron_getfield_arr(tmp_state, sub_field_names{sub_i});
                try tmp_state = cell2mat(tmp_state); catch, end
            end
        end
        out_cell = tmp_state;
%        % OLD EXTENDED CODE
%        for sub_i = 1:length(sub_field_names)
%             c_field_name = sub_field_names{sub_i};
%             tmp_cell = struct2cell(struct_arr);
%             names = fieldnames(struct_arr);
%             idx = find(strcmp(names, c_field_name));
%             if isempty(idx)
%                 error('Required field %s is not a field of input struct', c_field_name);
%             end
%             if length(idx) > 1
%                 error('Unknown bug.');
%             end
%             if sub_i<length(sub_field_names)
%                 struct_arr = [tmp_cell{idx, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :}];
%             end
%         out_cell(:, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :) = shiftdim( ...
%             tmp_cell(idx, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :), 1);
    end
end
