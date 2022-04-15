a = [1;2];
b = [-1;0];
c = [2;0];

x = [a(1), b(1), c(1)];
y = [a(2), b(2), c(2)];

v = [abs(x(3))+abs(x(2));abs(y(3))+abs(y(2))];
u = [abs(x(1))+abs(x(2));abs(y(1))+abs(y(2))];

proj = ((dot(u,v)/(norm(v)^2))* v) + b;

plotx = [x,a(1)];
ploty = [y,a(2)];

hold on
plot(plotx,ploty);
plot([a(1), proj(1)],[a(2), proj(2)], "r");
hold off
axis([-1 3 -1 3]);

