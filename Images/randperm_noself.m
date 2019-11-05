function p_noself = randperm_noself(m)
    % Unit test
    if nargin==0
        fprintf('randperm_noself: testing\n');
        for i = 1:10, disp(randperm_noself(5)); end
        for i = 1:10000, randperm_noself(500); end
        fprintf('randperm_noself: tests OK\n');
        return;
    end
    
    % Check inputs
    if length(m) ~= 1, error('Input must be length 1'); end
    if ~isnumeric(m), error('Input must be a number'); end
    if m == 1, error('Input must be larger than 1 - Can''t find a non-self permutation for length 1 input'); end
    
    % Go
    self = 1:m;
    p_start = randperm(m);
    p = p_start;
    p_self_idxs = find(p == self);
    p_self_idxs = p_self_idxs(randperm(length(p_self_idxs)));
    for i = 2:2:length(p_self_idxs)
        % switch pairs to take care of them
        idx_i1 = p_self_idxs(i-1);
        idx_i2 = p_self_idxs(i);
        p([idx_i1 idx_i2]) = p([idx_i2 idx_i1]);
    end
    if mod(length(p_self_idxs), 2)==1 % if odd still need to take care of last one
        self_idx = p_self_idxs(end);
        rand_idx = randi(m-1);
        if rand_idx >= self_idx, rand_idx = rand_idx + 1; end
        p([self_idx rand_idx]) = p([rand_idx self_idx]);
    end
    
    % Verify
	if any(p == self), error('Bug - self in permutation'); end
    if length(p) ~= m, error('Bug - wrong length'); end
    if length(p(p)) ~= length(p), error('Bug - non-permutation'); end
	p_noself = p;
end
