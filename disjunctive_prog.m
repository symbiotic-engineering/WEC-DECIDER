
%% Big M formulation
H = zeros(3);
H(1,2) = -1;
H(2,1) = -1;

xdotmax = 3;
Fmax = 5;
M = 1e3;
M2 = Fmax + 1;

A = [0 1 -M;
    0 -1 M;
    1 0 -1;
    1 0 M2];
b = [xdotmax; -xdotmax + M; Fmax; M2];

f = [0; 0; 0];

opts = optimoptions(@quadprog,'Algorithm','active-set');
x0 = [2 4 1];
[x, fval, exitflag, output] = quadprog(H,f,A,b,[],[],[],[],x0,opts)