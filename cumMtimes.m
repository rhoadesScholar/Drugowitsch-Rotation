function out = cumMtimes(vec)
    %cumulative product of matrices
    %make vec cell array of matrices
    %does not check to make sure dimensions agree
    out = NaN(size(vec{1},1), size(vec{1},2), length(vec));

    out(:,:,1) = vec{1};
    for i = 2:length(vec)
        out(:,:,i) = out(:,:,i-1)*vec{i};
    end

return
end