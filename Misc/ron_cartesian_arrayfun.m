function out = ron_cartesian_arrayfun(func, varargin)
    char_idx = find(cellfun(@ischar, varargin), 1, 'first');
    if isempty(char_idx), char_idx = length(varargin)+1; end
    vararg_to_cartesian = varargin(1:(char_idx-1));
    vararg_keep = varargin(char_idx:end);
    [vararg_cartesianed{1:length(vararg_to_cartesian)}] = ndgrid(vararg_to_cartesian{:});
    out = arrayfun(func, vararg_cartesianed{:}, vararg_keep{:});
end
