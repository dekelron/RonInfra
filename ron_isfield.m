% isfield which supports nested fields (separated by '.')
function out = ron_isfield(st, fn)
    fn_p = strsplit(fn, '.');
    if length(fn_p)==1
        out = isfield(st, fn);
    else
        for p_i = 1:length(fn_p)
            if ~isfield(st, fn_p{p_i})
                out = false;
                return;
            end
            st = st.(fn_p{p_i});
        end
        out = true;
    end
end