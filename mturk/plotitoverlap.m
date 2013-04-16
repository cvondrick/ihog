function plotit(what, resp, rgbresp),

clf;
if exist('rgbresp', 'var'),
  plot(resp.xxx, resp.percentoverlap, 'r-', 'LineWidth', 2);
  hold on;
  plot(resp.xxx, rgbresp.percentoverlap, 'g-', 'LineWidth', 2);
  plot([0 1], [0 0], 'k--', 'LineWidth', 2);
  set(legend('HOG+Human', 'RGB+Human', 'Chance'), 'FontSize', 15,'Location', 'SouthEast');
else,
  plot(resp.xxx, resp.percentoverlap, 'r-', 'LineWidth', 2);
  hold on;
  plot([0 1], [0 0], 'k--', 'LineWidth', 2);
  set(legend('HOG+Human', 'Chance'), 'FontSize', 15,'Location', 'SouthEast');
end

xlim([0 1]);
ylim([-1 1]);
xlabel('Detection Rank', 'FontSize', 25);
ylabel('Corr. Coeff. to DPM', 'FontSize', 25);
title(what, 'FontSize', 25);
grid on;
set(gcf, 'PaperPosition', [0 0 5 4]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 4]); %Set the paper to have width 5 and height 5.
