function logOdds = getLogOdds(MusLL)
    odds = @(a) [diff(a), diff(fliplr(a))];
    if size(MusLL) == 4
        temp = squeeze(MusLL(:,1:end-1,end,:));
        logOdds = NaN(size(temp));
        for i = 1:size(temp,1)
            logOdds(i,:,:) = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(i,:,j))), 1:size(temp,3), 'UniformOutput', false)), 1, size(temp,2), size(temp,3));
        end
    else
        temp = squeeze(MusLL(:,end,:));
        logOdds = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(:,j))), 1:size(temp,2), 'UniformOutput', false)), size(temp,1), size(temp,2));
    end
    
end