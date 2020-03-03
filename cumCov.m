function Vars = cumCov(data)
    %data colums are random variables
    %rows are observations
    temp = arrayfun(@(i) cov(data(1:i,:)), 2:size(data,1), 'UniformOutput', false);
    Vars = cat(3, eye(size(temp{1})), temp{:});
end