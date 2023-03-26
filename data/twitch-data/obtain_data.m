name_list = dir("./data_filtered_utc/*.txt");
name_len = length(name_list);

% obtain game list
game_list = [21779, 493057, 29595, 32399, 138585, 494717, 66170, ...
    33214, 497057, 493997, 496960, 498860, 495589, 488552, 18122, ...
    497439, 32982, 491487, 32959, 9234];
game_count = zeros(name_len,length(game_list) + 1);

% obtain contry list
language_list = {};
language_count = zeros(name_len, 100);


for ii = 1:1:name_len
    ii
    
    file_name = ['./data_filtered_utc/', name_list(ii).name];
    fidin=fopen(file_name);   
    
    while ~feof(fidin) 
        
        tline = fgetl(fidin);
        
        if tline
            tline_split = regexp(tline,'\t','split');
            tmp_language = tline_split{2};
            tmp_game = str2num(tline_split{3});

            % game 
            if tmp_game
                tmp_game_index = find(game_list == tmp_game);
                if tmp_game_index
                    game_count(ii, tmp_game_index) = game_count(ii, tmp_game_index) + 1;
                else
                    game_count(ii, end) = game_count(ii, end) + 1;
                end
            end
            
            % language
            if tmp_language
                if sum(ismember(language_list, tmp_language)) == 0
                    language_list{end+1} = tmp_language;
                    language_count(end) = language_count(end) + 1;
                else
                    tmp_language_index = find(ismember(language_list, tmp_language) == 1);
                    language_count(ii, tmp_language_index) = language_count(ii, tmp_language_index) + 1;
                end
            end
            
        end
    end

end

% game

[value, index] = sort(sum(game_count(:,1:20)),'descend');
game_count_first5 = zeros(name_len, 6);
game_count_first5(:,1:5) = game_count(:,index(1:5));
game_count_first5(:,6) = sum(game_count(:,index(6:end)),2) + game_count(:,end);

figure;
mean_value = mean(game_count_first5(:,end:-1:1));
bar_game_data = [mean_value./sum(mean_value); game_count_first5(1:4:40,end:-1:1) ./ sum(game_count_first5(1:4:40,end:-1:1),2)];
bar([0.5, 2:1:11],bar_game_data,'stacked')
labels={'others', game_list(index(5)),game_list(index(4)),game_list(index(3)),...
    game_list(index(2)),game_list(index(1))};
fliplegend(labels);

hold on
plot([-0.5, 12], [mean_value(1)/sum(mean_value), mean_value(1)/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:2))/sum(mean_value), sum(mean_value(1:2))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:3))/sum(mean_value), sum(mean_value(1:3))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:4))/sum(mean_value), sum(mean_value(1:4))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:5))/sum(mean_value), sum(mean_value(1:5))/sum(mean_value)]);




% language
language_count = language_count(:, 1:30);
[value, index] = sort(sum(language_count),'descend');

language_count_first5 = zeros(name_len, 6);
language_count_first5(:,1:5) = language_count(:,index(1:5));
language_count_first5(:,6) = sum(language_count(:,index(6:end)),2);

figure;
mean_value = mean(language_count_first5(:,end:-1:1));
bar_language_data = [mean_value./sum(mean_value); language_count_first5(1:4:40,end:-1:1) ./ sum(language_count_first5(1:4:40,end:-1:1),2)];
bar([0.5, 2:1:11],bar_language_data,'stacked')
labels={'others', language_list{index(5)},language_list{index(4)},language_list{index(3)},...
    language_list{index(2)},language_list{index(1)}, };
fliplegend(labels);

hold on
plot([-0.5, 12], [mean_value(1)/sum(mean_value), mean_value(1)/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:2))/sum(mean_value), sum(mean_value(1:2))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:3))/sum(mean_value), sum(mean_value(1:3))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:4))/sum(mean_value), sum(mean_value(1:4))/sum(mean_value)]);
plot([-0.5, 12], [sum(mean_value(1:5))/sum(mean_value), sum(mean_value(1:5))/sum(mean_value)]);

%{
file_name = ['./data_filtered_utc/', name_list(1).name];
fidin=fopen(file_name);   
    
while ~feof(fidin) 
        
    tline = fgetl(fidin);
    if tline
        tline_split = regexp(tline,'\t','split');
        tmp_language = tline_split{2};

        if sum(ismember(language_list, tmp_language)) == 0
            language_list{end+1} = tmp_language;
        end
    end
end

language_list
%}


