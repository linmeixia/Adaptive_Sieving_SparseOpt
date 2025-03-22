function [group_info_reduced] = generate_group(idx,group_info)
org_group = group_info.org_group;
group = org_group(idx);
[s,P_reduced] = sort(group);
[~,uidx] = unique(s);
group_num_reduced = length(uidx);
M_reduced = zeros(2,group_num_reduced);
org_group_reduced = zeros(length(idx),1);
for i = 1:group_num_reduced
    if i ~= group_num_reduced
        M_reduced(:,i) = [uidx(i);uidx(i+1)-1];
    else
        M_reduced(:,i) = [uidx(i);length(idx)];
    end
    org_group_reduced(P_reduced(M_reduced(1,i):M_reduced(2,i))) = i;
end
group_info_reduced.M = M_reduced(:,1:group_num_reduced);
group_info_reduced.P = P_reduced;
[~,group_info_reduced.PT] = sort(P_reduced);
group_info_reduced.org_group = org_group_reduced;
end



% function [group_info_reduced] = generate_group(idx,n_reduced,group_info)
% M = group_info.M;
% P = group_info.P;
% group_num = size(M,2);
% M_reduced = zeros(size(M));
% group_num_reduced = 0;
% P_reduced = zeros(1,n_reduced);
% for i = 1:group_num
%     [~,idxtmp] = intersect(idx,P(M(1,i):M(2,i)));
%     if ~isempty(idxtmp)
%         group_num_reduced = group_num_reduced+1;
%         lengthtmp = length(idxtmp);
%         if group_num_reduced == 1
%             tmp2 = [1;lengthtmp];
%             P_reduced(1:lengthtmp) = idxtmp;
%         else
%             tmp2 = [M_reduced(2,group_num_reduced-1)+1;M_reduced(2,group_num_reduced-1)+lengthtmp];
%             P_reduced(M_reduced(2,group_num_reduced-1)+1:M_reduced(2,group_num_reduced-1)+lengthtmp) = idxtmp;
%         end
%         M_reduced(:,group_num_reduced) = tmp2;
%     end
% end
% group_info_reduced.M = M_reduced(:,1:group_num_reduced);
% group_info_reduced.P = P_reduced;
% [~,group_info_reduced.PT] = sort(P_reduced);
% end

