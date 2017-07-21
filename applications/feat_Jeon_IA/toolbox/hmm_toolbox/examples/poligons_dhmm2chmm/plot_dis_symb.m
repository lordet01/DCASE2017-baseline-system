% This function was created to plot the distribution of synbols for a discrete HMM
% In particular, we use the handler of the figure where we want to plot it.

function plot_dis_symb(class,group,parameter,number)

load hmmpoligonos
 
 for state=1:10
     prob=B{class,group}{parameter}(state,:);
 if (state<6)
     figure(3)
 subplot(2,5,state+5);
 plot(prob);
 axis([1 length(prob) 0 max(prob)]);
 title(['symbol prob.']);
else 
    figure(4);
 subplot(2,5,state);
 plot(prob);
  axis([1 length(prob) 0 max(prob)]);
  title(['symbol prob.']);
end
end