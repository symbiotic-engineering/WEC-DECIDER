close all
t = linspace(0,2*pi);
x = sin(t);
xdot = cos(t);

a_vals = [0,0.25,0.5,0.75,1,2];
ymax = max(a)+1;

figure
hold on

% shading to show motor vs generator
patch([1,1,0,0],ymax*[0,1,1,0],'g','FaceAlpha',0.2,'DisplayName','Generating');
patch([-1,-1,0,0],ymax*[0,1,1,0],'r','FaceAlpha',0.2,'DisplayName','Motoring');
patch([-1,-1,0,0],ymax*[0,-1,-1,0],'g','FaceAlpha',0.2,'HandleVisibility','off')
patch([1,1,0,0],ymax*[0,-1,-1,0],'r','FaceAlpha',0.2,'HandleVisibility','off')

% change colors to be rainbow ordered
cols = get(gca,'ColorOrder');
new_cols = cols([2 3 5 6 1 4],:);
colororder(new_cols);

% torque speed curves
for a = a_vals
    T = xdot + a * x;
    plot(xdot, T,'DisplayName',num2str(a))
end

% dashed axis lines
plot([-1,1], [0,0], 'k--', [0,0], ymax*[-1,1], 'k--','HandleVisibility','off','LineWidth',3)

plots = get(gca,'Children');
leg = legend(plots([length(a_vals):-1:1, end, end-1])); % put the shading legend entries at the bottom
xlabel('\Omega')
ylabel('\tau')
title('Notional Torque Speed Curve: $\tau = Z_{gen} \Omega$','Interpreter','latex')
title(leg,'$Im(Z_{gen})/Re(Z_{gen})$','Interpreter','latex')
improvePlot

%xlim([0 1])
%ylim([0 max(a)+1])