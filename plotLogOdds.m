function plotLogOdds(MusLL, allT, labels)
    logOdds = getLogOdds(MusLL);
    colors = linspecer(size(logOdds,1)*size(logOdds,2));
    figure;
    for i = 1:size(logOdds,1)
       for j = 1:size(logOdds,2)
            plot(squeeze(logOdds(i,j,:)), allT, 'LineWidth',2, 'Color', colors((j-1)*size(logOdds,2) + i, :), 'DisplayName', sprintf('W_{%s}: M_{%s}/M_{%s}', labels{1, j}, labels{2, :}))%will only work for 2 model case
            hold on
            xlabel("time")
            ylabel("Log Odds Ratio")
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            title("Odds of Models")
            legend
       end
    end
    return
end