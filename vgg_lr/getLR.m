function [ slic_S_value ] = getLR( DI,F_count,slic_L,slic_N)
DI_Reshape = reshape(DI,[],F_count);
SLIC_Feature_Map = zeros(slic_N,F_count);
    
for i = 1:slic_N
    block_index = find(slic_L == i - 1);
    f = DI_Reshape(block_index,:);
    f = mean(f);
    SLIC_Feature_Map(i,:) = f;
end

lamda = double(slic_N)^(-1/2);
[L,S,Y] = singular_value_rpca(SLIC_Feature_Map,lamda);

S_norm_1 = zeros(slic_N,1);
for i = 1:slic_N
    S_norm_1(i) = norm(S(i,:),1);
end

[v,index] = sort(S_norm_1,'descend');
slic_S = zeros(size(slic_L));
slic_S_value = zeros(size(slic_L));
for i = 1:slic_N
    block_index = find(slic_L == (index(i) - 1));
    slic_S(block_index) = slic_N - i;
    slic_S_value(block_index) = v(i);
end
end


