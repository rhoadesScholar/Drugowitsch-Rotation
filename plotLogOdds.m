function plotLogOdds(MusLL, allT, labels)
    logOdds = getLogOdds(MusLL);
    colors = linspecer(size(logOdds,1)*size(logOdds,2));
    figure;
    for i = 1:size(logOdds,1)
       for j = 1:size(logOdds,2)
            plot(allT, squeeze(logOdds(i,j,:)), 'LineWidth',2, 'Color', colors((j-1)*size(logOdds,2) + i, :), 'DisplayName', sprintf('W_{%s}: M_{%s}/M_{%s}', labels{1, j}, labels{2, 1}, labels{2, 2}))%will only work for 2 model case
            hold on
            xlabel("time")
            ylabel("Log Odds Ratio")
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            title("Odds of Models")
            legend('Location', 'best')
       end
    end
    return
end