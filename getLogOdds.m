function logOdds = getLogOdds(MusLL)
%     odds = @(a) [diff(a), diff(fliplr(a))];
    odds = @(a) diff(a);
    if length(size(MusLL)) == 4
        temp = MusLL(:,1:end-1,end,:);
        temp = reshape(temp, size(temp,1), size(temp,2), []);
%         logOdds = NaN(size(temp));
        logOdds = NaN(size(temp,1), size(temp,2)-1, size(temp,3));
        for i = 1:size(temp,1)
            logOdds(i,:,:) = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(i,:,j))), 1:size(temp,3), 'UniformOutput', false)), 1, size(temp,2)-1, size(temp,3));
        end
    else
        temp = squeeze(MusLL(:,end,:));
        logOdds = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(:,j))), 1:size(temp,2), 'UniformOutput', false)), size(temp,1), size(temp,2));
    end
    
end