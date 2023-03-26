noniid = 95;
lr = 30000;
e = 5;
g = 5;
method = 'fedprox';
num = 5;

epoch = 200;
fl_method = 'mnist';
% FedAvg ===================================================

loss = zeros(epoch, num);
train = zeros(epoch, num);

for ii = 0: 1: num - 1
    
     file_name = [fl_method, '_', method, '_100_straFalse_None_n100f10e', ...
         num2str(e),'b50g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt'];
    
     
     [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
     loss(:,ii + 1) = tmp_test_loss(1:epoch);
     train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end

if num == 1
    avg_loss = loss;
    avg_train = train;
else
    avg_loss = mean(loss, 2);
    avg_train = mean(train, 2);
end


% Optimal ===================================================
loss = zeros(epoch, num);
train = zeros(epoch, num);

for ii = 0: 1: num - 1
        
    file_name = [fl_method, '_', method, '_100_straTrue_optimal_n100f10e', ...
         num2str(e),'b50g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt']; 
     
    [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
    loss(:,ii + 1) = tmp_test_loss(1:epoch);
    train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end

if num == 1
    opt_loss = loss;
    opt_train = train;
else
    opt_loss = mean(loss, 2);
    opt_train = mean(train, 2);
end

% Proportional ===================================================
loss = zeros(epoch, num);
train = zeros(epoch, num);

for ii = 0: 1: num - 1
    
    
    file_name = [fl_method, '_', method, '_100_straTrue_proportional_n100f10e', ...
         num2str(e),'b50g', num2str(g), 'noniid', num2str(noniid), 'lr', ...
         num2str(lr), '_', num2str(ii), '.txt']; 
     
    [tmp_train_loss,tmp_train_acc,tmp_test_loss,tmp_test_acc] = textread(file_name,'%f%f%f%f');
    loss(:,ii + 1) = tmp_test_loss(1:epoch);
    train(:, ii + 1) = tmp_test_acc(1:epoch);
     
end


if num == 1
    pp_loss = loss;
    pp_train = train;
else
    pp_loss = mean(loss, 2);
    pp_train = mean(train, 2);
end


figure;
plot(smoothdata(avg_loss,'movmedian', 10));
set(gca,'YScale','log')
hold on
plot(smoothdata(opt_loss,'movmedian', 10));
plot(smoothdata(pp_loss,'movmedian', 10));
plot([0,200],[1.5, 1.5])