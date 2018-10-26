file_dados = dlmread('X_19.csv');
file_dados(1,:) = [];
dim = size(file_dados);
file_amostras = file_dados(:,1:18);
fim_normal = 60;
inicio_falha = 61;
falha = 19;
file_amostras_normal = file_amostras(1:fim_normal,:);
mean_normal = mean(file_amostras(1:fim_normal,:));
sd_normal = std(file_amostras(1:fim_normal,:));
file_amostras_falha = file_amostras(inicio_falha:end,:);
dim_1 = size(file_amostras_falha);
file_amostras_normal = bsxfun(@minus,file_amostras_normal,mean_normal);
file_amostras_normal = bsxfun(@rdivide,file_amostras_normal,sd_normal);
file_amostras_normal = center(file_amostras_normal);

file_label = file_dados(:,19);
file_label(fim_normal+1:end) = falha;

file_amostras_falha = bsxfun(@minus,file_amostras_falha,mean_normal);
file_amostras_falha = bsxfun(@rdivide,file_amostras_falha,sd_normal);

file_amostras_reunido = file_amostras_normal;
file_amostras_reunido(fim_normal+1:dim(1),:) = file_amostras_falha;

%pos = find(file_label==0);
%neg = find(file_label==falha);
%plot3(file_amostras_reunido(pos,1),file_amostras_reunido(pos,2),file_amostras_reunido(pos,3),"o", file_amostras_reunido(neg,1),file_amostras_reunido(neg,2),file_amostras_reunido(neg,3), "+")

cv = cov(file_amostras_normal); 
[v, lambda] = eig(cv); 
[sorteigen, order] = sort(diag(lambda),'descend');
v = fliplr(v);
lambda = fliplr(lambda);
lambda = flipud(lambda);
varian_compo = cumsum(lambda) / sum(lambda);
prc = sum( varian_compo <= 0.95);
v_1 = v(:,1:prc);
lambda_1 = lambda(:,1:prc);

file_amostras_trans = file_amostras_reunido * v_1;
file_amostras_trans(:,prc+1) = file_label;

file_amostras_trans_visualização = file_amostras_trans;
file_amostras_trans_visualização(61:end,:) = log(file_amostras_trans(61:end,:));
file_amostras_trans_visualização(:,prc+1) = file_label;
pos = find(file_label==0);
neg = find(file_label==falha);
figure()
plot(file_amostras_trans_visualização(pos,1),file_amostras_trans_visualização(pos,2),"o", file_amostras_trans_visualização(neg,1),file_amostras_trans_visualização(neg,2), "+")
xlabel("PCA1")
ylabel("PCA2")
title("Visualizacao componentes principais");
figure()
plot3(file_amostras_trans_visualização(pos,1),file_amostras_trans_visualização(pos,2),file_amostras_trans_visualização(pos,3),"o", file_amostras_trans_visualização(neg,1),file_amostras_trans_visualização(neg,2),file_amostras_trans_visualização(neg,3), "+")
xlabel("PCA1")
ylabel("PCA2")
zlabel("PCA3")
title("Visualizacao componentes principais");

%Calculando o T2
t2limit = ((prc*(fim_normal-1))/ (fim_normal - prc)) * finv(0.95,60,11); 
T2 = zeros(101,1);
lambda_2 = lambda_1(1:prc,:);
lambda_2 = inv(lambda_2)
for i = 1:101
   T2(i) = (file_amostras_trans(i,1:prc) * lambda_2) * file_amostras_trans(i,1:prc)';
endfor

figure()
plot(log(T2));
axis(limits=[0,101])
hold on
plot([0,101],[log(t2limit),log(t2limit)])
xlabel("Tempo");
ylabel("Valor T2");
title("Visualizacao T2");