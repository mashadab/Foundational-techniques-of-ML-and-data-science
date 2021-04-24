%cmdscale algorithm
%Mohammad Afzal Shadab
%mashadab@utexas.edu
%04/13/2021

clear all; clc;

%% Step 1 - Forming the distance matrix XX^T
A = importdata('us_distance.txt');
% A is lower triangular, let's make it symmetric
D = A + A.';
n = size(D,1);

%Cities
cities = {'BOS','BUF','CHI','DAL','DEN','HOU','LA','MEM',...
    'MIA','MIN','NYC','OMA','PHI','PHO','PIT',...
    'SLO','SLC', 'SF', 'SEA','DC'};

%% Method 1: Using inbuilt MATLAB function cmdscale

[X,e] = cmdscale(D,2);

figure(1);
hold on
scatter(-X(:,1),-X(:,2), 'filled')
text(-X(:,1)+0.1,-X(:,2)+0.1,cities)
title('Using cmdscale')
xlabel('x [Miles]');
ylabel('y [Miles]');
%set(gca,'visible','off')

%% Method 2: Using our approach
%Computing Cov(X)=XX^T
Cov = zeros(size(D));
for i=1:n
    for j = 1:n
        Cov(i,j) = -0.5*(D(i,j)^2 - norm(D(:,i))^2/n - norm(D(:,j))^2/n + norm(D,'fro')/n^2); 
    end
end

%% Part 2: Perform the SVD of Cov(X)
[U, S, ~] = svd(Cov);


%% Part 3: Using best rank k=2 approximation for X
%Rotate the data using V, which is arbitrarity chosed

theta = pi;
V = [cos(theta), -sin(theta); sin(theta), cos(theta)];
X = U(:,2:3)*sqrt(S(2:3,2:3))*V.'  %First singular value refers to the mean

figure(2);
hold on
scatter(X(:,1),X(:,2), 'filled')
title('Present algorithm')
text(X(:,1)+0.1,X(:,2)+0.1,cities)
xlabel('x [Miles]');
ylabel('y [Miles]');

    