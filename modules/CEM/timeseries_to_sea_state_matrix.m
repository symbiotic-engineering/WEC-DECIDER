function [carbon_contour,price_contour] = timeseries_to_sea_state_matrix(Hs,T,jpd_Hs,jpd_T,carbon_interpolated,price_interpolated)

%make the boundary for the first bucket 0
p.Hs = [0; jpd_Hs];
p.T = [0, jpd_T];

%find the average price or carbon value for a slice on the xy plane
price_contour=zeros(length(p.Hs)-1,length(p.T)-1);
carbon_contour=zeros(length(p.Hs)-1,length(p.T)-1);

for i = 1:length(p.Hs)-1
    for j = 1:length(p.T)-1
        idx_use = Hs > p.Hs(i) & Hs < p.Hs(i+1) & T > p.T(j) & T < p.T(j+1);
        price_contour(i,j) = mean(price_interpolated(idx_use),'all');
        carbon_contour(i,j) = mean(carbon_interpolated(idx_use),'all');
    end
end

%make contour plots
[T_mdocean,Hs_mdocean] = meshgrid(p.T(2:end),p.Hs(2:end));
figure
contourf(Hs_mdocean, T_mdocean, price_contour)
xlabel('H_{s} (m)')
ylabel('T (s)')
title('Price')
colorbar
figure
contourf(Hs_mdocean, T_mdocean, carbon_contour)
title('Carbon')
xlabel('H_{s} (m)')
ylabel('T (s)')
colorbar