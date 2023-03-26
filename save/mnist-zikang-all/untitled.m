noniid = 95;
lr = 30000;
e = 10;
g = 5;
method = 'fedavg';

epoch = 200;

% FedAvg ===================================================
loss = zeros(200, 4);
train = zeros(200, 4);

for ii = 0:1:4
    
     file_name = ['./e', num2str(e), '/mnist_', method, '_100_straFalse_None_n100f10e', ...
         num2str(e),'b10g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt'];
    
     
     [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
     loss(:,ii + 1) = tmp_test_loss(1:epoch);
     train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end

avg_loss = mean(loss, 2);
avg_train = mean(train, 2);


% Optimal ===================================================
loss = zeros(200, 4);
train = zeros(200, 4);

for ii = 0:1:4
    
     file_name = ['./e', num2str(e), '/mnist_', method, '_100_straTrue_optimal_n100f10e', ...
         num2str(e),'b10g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt'];
    
     
     [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
     loss(:,ii + 1) = tmp_test_loss(1:epoch);
     train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end

opt_loss = mean(loss, 2);
opt_train = mean(train, 2);


% Proportional ===================================================
loss = zeros(200, 4);
train = zeros(200, 4);

for ii = 0:1:4
    
     file_name = ['./e', num2str(e), '/mnist_', method, '_100_straTrue_proportional_n100f10e', ...
         num2str(e),'b10g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt'];
    
     
     [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
     loss(:,ii + 1) = tmp_test_loss(1:epoch);
     train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end

pp_loss = mean(loss, 2);
pp_train = mean(train, 2);



figure;
plot(smoothdata(avg_loss,'movmedian', 10));
set(gca,'YScale','log')
hold on
plot(smoothdata(opt_loss,'movmedian', 10));
plot(smoothdata(pp_loss,'movmedian', 10));