function plotMSE(MusLL, dims, Vars, allT, labels)
    [MSE, mVars] = getMSE(MusLL, dims, Vars);
    width = 2;
    
    colors = linspecer(size(MSE,1)*size(MSE,2));
    figure; 
    for i = 1:size(MSE,2)
        for j = 1:size(MSE,1)
            subplot(2, 2, 1)
            plot(allT, squeeze(MSE(j, i, 1, :)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'DisplayName', sprintf('W_{%s}: M_{%s}', labels{1, j}, labels{2, i}))
            hold on
            if j == size(MSE, 1) && ~contains(labels{2, i}, 'Combined') && ~contains(labels{2, i}, 'Decided')
                plot(allT, squeeze(mVars(i, 1,:)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: M_{%s}', labels{2, i}));
            end
            xlabel("time")
            ylabel("mean square error")
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            legend('Location', 'best')
            title("Position MSE")

            if size(MSE,3) > 3
                subplot(2, 2, 2)
                plot(allT, squeeze(MSE(j, i, 2, :)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'DisplayName', sprintf('W_{%s}: M_{%s}', labels{1, j}, labels{2, i}))
                hold on
                if j == size(MSE, 1) && ~contains(labels{2, i}, 'Combined') && ~contains(labels{2, i}, 'Decided')
                    plot(allT, squeeze(mVars(i, 2,:)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: M_{%s}', labels{2, i}));
                end
                xlabel("time")
                ylabel("mean square error")
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                title("Object Distance MSE")
            end

            subplot(2, 2, 3)
            plot(allT, squeeze(MSE(j, i, end-1, :)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'DisplayName', sprintf('W_{%s}: M_{%s}', labels{1, j}, labels{2, i}))
            hold on
            if j == size(MSE, 1) && ~contains(labels{2, i}, 'Combined') && ~contains(labels{2, i}, 'Decided')
                plot(allT, squeeze(mVars(i, end,:)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: M_{%s}', labels{2, i}));
            end
            xlabel("time")
            ylabel("mean square error")
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            title("Velocity MSE")

            subplot(2, 2, 4)
            plot(allT, squeeze(MSE(j, i, end, :)), 'Color', colors((j-1)*size(MSE,2) + i, :), 'LineWidth', width, 'DisplayName', sprintf('W_{%s}: M_{%s}', labels{1, j}, labels{2, i}))
            hold on
            xlabel("time")
            ylabel("Log Evidence")
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            title("Model Performance")
        end
    end
    return
end

function [MSE, MVar] = getMSE(MusLL, dims, Vars)  
    MVar = NaN(size(Vars,4), size(Vars,1), size(Vars,3));
    for i = 1:size(Vars,4)
        MVar(i,:,:) = cell2mat(arrayfun(@(t) diag(Vars(:,:,t,i)), 1:size(Vars,3), 'UniformOutput', false));
    end
    MVar = nansum(reshape(MVar, size(MVar,1), dims, [], size(MVar,3)), 2);
    MVar = reshape(MVar, size(MVar,1), [], size(MVar,4));
    
    MSE = nansum(reshape(MusLL(:,:,1:end-1,:), size(MusLL,1), size(MusLL,2), dims, [], size(MusLL,4)), 3);
    MSE = reshape(MSE, size(MusLL,1), size(MusLL,2), [], size(MusLL,4));
    try
        MSE = cat(3, MSE, MusLL(:,:,end,:));
    catch ME
        if strcmp(ME.identifier,'MATLAB:catenate:dimensionMismatch')
            try
                MSE = cat(3, nansum(squeeze(reshape(MusLL(:,:,1:end-1,:), size(MusLL,1), size(MusLL,2), dims, [], size(MusLL,4))), 3), MusLL(:,:,end,:));
            catch
               beep 
            end
        end
    end
    
    return
end