function plotit(resp, rgbresp),

clf;

if exist('rgbresp', 'var'),
  plot(resp.hoggles.rec, resp.hoggles.prec, 'r-', 'LineWidth', 2);
  hold on;
  plot(rgbresp.hoggles.rec, rgbresp.hoggles.prec, 'g-', 'LineWidth', 2);
  plot(resp.dpm.rec, resp.dpm.prec, 'b-', 'LineWidth', 2);
  set(legend(sprintf('HOG+Human AP = %0.2f', resp.hoggles.ap), sprintf('RGB+Human AP = %0.2f', rgbresp.hoggles.ap), sprintf('HOG+DPM AP = %0.2f', resp.dpm.ap)), 'FontSize', 15,'Location', 'SouthWest');
else,
  plot(resp.hoggles.rec, resp.hoggles.prec, 'r-', 'LineWidth', 2);
  hold on;
  plot(resp.dpm.rec, resp.dpm.prec, 'b-', 'LineWidth', 2);
  set(legend(sprintf('HOG+Human AP = %0.2f', resp.hoggles.ap), sprintf('HOG+DPM AP = %0.2f', resp.dpm.ap)), 'FontSize', 15,'Location', 'SouthWest');
end

xlim([0 1]);
ylim([0 1]);
xlabel('Recall', 'FontSize', 25);
ylabel('Precision', 'FontSize', 25);
title(sprintf('Person'), 'FontSize', 25);
grid on;
set(gcf, 'PaperPosition', [0 0 5 4]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 4]); %Set the paper to have width 5 and height 5.
